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

module RV_GEMM_regslice_both
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

module RV_GEMM_regslice_both_w1
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

module RV_GEMM_rv_gemm_one_layer (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        mode_val,
        pos_r,
        rq_stream_TDATA,
        rq_stream_TVALID,
        rq_stream_TREADY,
        rs_stream_TDATA,
        rs_stream_TVALID,
        rs_stream_TREADY,
        v_stream_TDATA,
        v_stream_TVALID,
        v_stream_TREADY,
        vq_cache_i_stream_TDATA,
        vq_cache_i_stream_TVALID,
        vq_cache_i_stream_TREADY,
        vs_cache_i_stream_TDATA,
        vs_cache_i_stream_TVALID,
        vs_cache_i_stream_TREADY,
        vq_cache_o_stream_TDATA,
        vq_cache_o_stream_TVALID,
        vq_cache_o_stream_TREADY,
        vs_cache_o_stream_TDATA,
        vs_cache_o_stream_TVALID,
        vs_cache_o_stream_TREADY,
        aq_stream_TDATA,
        aq_stream_TVALID,
        aq_stream_TREADY,
        as_stream_TDATA,
        as_stream_TVALID,
        as_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 16'd1;
parameter    ap_ST_fsm_state2 = 16'd2;
parameter    ap_ST_fsm_state3 = 16'd4;
parameter    ap_ST_fsm_state4 = 16'd8;
parameter    ap_ST_fsm_state5 = 16'd16;
parameter    ap_ST_fsm_state6 = 16'd32;
parameter    ap_ST_fsm_state7 = 16'd64;
parameter    ap_ST_fsm_state8 = 16'd128;
parameter    ap_ST_fsm_state9 = 16'd256;
parameter    ap_ST_fsm_state10 = 16'd512;
parameter    ap_ST_fsm_state11 = 16'd1024;
parameter    ap_ST_fsm_state12 = 16'd2048;
parameter    ap_ST_fsm_state13 = 16'd4096;
parameter    ap_ST_fsm_state14 = 16'd8192;
parameter    ap_ST_fsm_state15 = 16'd16384;
parameter    ap_ST_fsm_state16 = 16'd32768;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [0:0] mode_val;
input  [31:0] pos_r;
input  [63:0] rq_stream_TDATA;
input   rq_stream_TVALID;
output   rq_stream_TREADY;
input  [7:0] rs_stream_TDATA;
input   rs_stream_TVALID;
output   rs_stream_TREADY;
input  [183:0] v_stream_TDATA;
input   v_stream_TVALID;
output   v_stream_TREADY;
input  [63:0] vq_cache_i_stream_TDATA;
input   vq_cache_i_stream_TVALID;
output   vq_cache_i_stream_TREADY;
input  [7:0] vs_cache_i_stream_TDATA;
input   vs_cache_i_stream_TVALID;
output   vs_cache_i_stream_TREADY;
output  [63:0] vq_cache_o_stream_TDATA;
output   vq_cache_o_stream_TVALID;
input   vq_cache_o_stream_TREADY;
output  [7:0] vs_cache_o_stream_TDATA;
output   vs_cache_o_stream_TVALID;
input   vs_cache_o_stream_TREADY;
output  [63:0] aq_stream_TDATA;
output   aq_stream_TVALID;
input   aq_stream_TREADY;
output  [7:0] as_stream_TDATA;
output   as_stream_TVALID;
input   as_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg rq_stream_TREADY;
reg rs_stream_TREADY;
reg v_stream_TREADY;
reg vq_cache_i_stream_TREADY;
reg vs_cache_i_stream_TREADY;
reg[63:0] vq_cache_o_stream_TDATA;
reg vq_cache_o_stream_TVALID;
reg[7:0] vs_cache_o_stream_TDATA;
reg vs_cache_o_stream_TVALID;
reg aq_stream_TVALID;
reg as_stream_TVALID;

(* fsm_encoding = "none" *) reg   [15:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [0:0] tmp_reg_1566;
reg   [9:0] trunc_ln78_1_reg_1572;
wire   [0:0] mode_val_read_read_fu_452_p2;
wire    ap_CS_fsm_state2;
wire   [9:0] decoded_chunk_fu_747_p3;
reg   [9:0] decoded_chunk_reg_1581;
wire   [30:0] add_ln86_fu_758_p2;
reg   [30:0] add_ln86_reg_1593;
wire   [30:0] valid_s_fu_774_p3;
reg   [30:0] valid_s_reg_1605;
wire    ap_CS_fsm_state3;
reg   [7:0] trunc_ln102_2_reg_1611;
wire   [7:0] valid_st_fu_845_p3;
reg   [7:0] valid_st_reg_1616;
wire    ap_CS_fsm_state4;
wire   [2:0] kv_2_fu_862_p2;
reg   [2:0] kv_2_reg_1625;
wire    ap_CS_fsm_state5;
wire   [3:0] hct_1_fu_874_p2;
reg   [3:0] hct_1_reg_1633;
wire    ap_CS_fsm_state7;
wire   [2:0] trunc_ln194_fu_880_p1;
reg   [2:0] trunc_ln194_reg_1638;
wire   [1:0] r_1_fu_1146_p2;
reg   [1:0] r_1_reg_1838;
wire    ap_CS_fsm_state11;
reg    vq_buf_ce0;
wire   [63:0] vq_buf_q0;
reg   [9:0] vq_buf_address1;
reg    vq_buf_ce1;
reg   [7:0] vq_buf_we1;
reg   [63:0] vq_buf_d1;
reg    vq_buf_1_ce0;
wire   [63:0] vq_buf_1_q0;
reg   [9:0] vq_buf_1_address1;
reg    vq_buf_1_ce1;
reg   [7:0] vq_buf_1_we1;
reg   [63:0] vq_buf_1_d1;
reg    vq_buf_2_ce0;
wire   [63:0] vq_buf_2_q0;
reg   [9:0] vq_buf_2_address1;
reg    vq_buf_2_ce1;
reg   [7:0] vq_buf_2_we1;
reg   [63:0] vq_buf_2_d1;
reg    vq_buf_3_ce0;
wire   [63:0] vq_buf_3_q0;
reg   [9:0] vq_buf_3_address1;
reg    vq_buf_3_ce1;
reg   [7:0] vq_buf_3_we1;
reg   [63:0] vq_buf_3_d1;
reg    vq_buf_4_ce0;
wire   [63:0] vq_buf_4_q0;
reg   [9:0] vq_buf_4_address1;
reg    vq_buf_4_ce1;
reg   [7:0] vq_buf_4_we1;
reg   [63:0] vq_buf_4_d1;
reg    vq_buf_5_ce0;
wire   [63:0] vq_buf_5_q0;
reg   [9:0] vq_buf_5_address1;
reg    vq_buf_5_ce1;
reg   [7:0] vq_buf_5_we1;
reg   [63:0] vq_buf_5_d1;
reg    vq_buf_6_ce0;
wire   [63:0] vq_buf_6_q0;
reg   [9:0] vq_buf_6_address1;
reg    vq_buf_6_ce1;
reg   [7:0] vq_buf_6_we1;
reg   [63:0] vq_buf_6_d1;
reg    vq_buf_7_ce0;
wire   [63:0] vq_buf_7_q0;
reg   [9:0] vq_buf_7_address1;
reg    vq_buf_7_ce1;
reg   [7:0] vq_buf_7_we1;
reg   [63:0] vq_buf_7_d1;
reg   [9:0] vs_buf_address0;
reg    vs_buf_ce0;
wire   [31:0] vs_buf_q0;
reg   [9:0] vs_buf_address1;
reg    vs_buf_ce1;
reg    vs_buf_we1;
reg   [31:0] vs_buf_d1;
wire    grp_load_llm_v_cache_qs_fu_480_ap_start;
wire    grp_load_llm_v_cache_qs_fu_480_ap_done;
wire    grp_load_llm_v_cache_qs_fu_480_ap_idle;
wire    grp_load_llm_v_cache_qs_fu_480_ap_ready;
wire    grp_load_llm_v_cache_qs_fu_480_vq_cache_i_stream_TREADY;
wire    grp_load_llm_v_cache_qs_fu_480_vs_cache_i_stream_TREADY;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_0_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_0_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_0_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_0_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_1_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_1_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_1_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_1_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_2_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_2_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_2_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_2_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_3_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_3_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_3_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_3_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_4_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_4_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_4_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_4_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_5_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_5_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_5_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_5_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_6_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_6_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_6_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_6_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_q_buf_7_address1;
wire    grp_load_llm_v_cache_qs_fu_480_q_buf_7_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_fu_480_q_buf_7_we1;
wire   [63:0] grp_load_llm_v_cache_qs_fu_480_q_buf_7_d1;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_s_buf_address0;
wire    grp_load_llm_v_cache_qs_fu_480_s_buf_ce0;
wire   [9:0] grp_load_llm_v_cache_qs_fu_480_s_buf_address1;
wire    grp_load_llm_v_cache_qs_fu_480_s_buf_ce1;
wire    grp_load_llm_v_cache_qs_fu_480_s_buf_we1;
wire   [31:0] grp_load_llm_v_cache_qs_fu_480_s_buf_d1;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_done;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_idle;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_ready;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_v_stream_TREADY;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i138_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i138_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i136_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i136_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i134_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i134_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i132_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i132_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i130_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i130_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i128_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i128_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i126_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i126_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i124_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i124_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i122_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i122_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i120_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i120_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i118_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i118_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i116_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i116_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i114_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i114_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i112_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i112_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i110_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i110_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i108_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i108_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i106_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i106_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i104_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i104_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i102_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i102_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i100_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i100_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i98_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i98_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i96_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i96_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i94_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i94_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i92_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i92_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i90_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i90_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i88_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i88_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i86_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i86_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i84_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i84_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i82_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i82_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i80_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i80_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i78_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i78_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i76_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i76_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i74_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i74_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i72_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i72_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i70_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i70_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i68_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i68_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i66_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i66_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i64_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i64_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i62_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i62_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i60_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i60_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i58_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i58_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i56_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i56_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i54_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i54_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i52_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i52_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i50_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i50_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i48_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i48_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i46_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i46_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i44_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i44_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i42_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i42_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i40_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i40_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i38_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i38_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i36_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i36_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i34_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i34_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i32_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i32_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i30_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i30_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i28_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i28_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i26_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i26_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i24_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i24_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i22_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i22_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i20_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i20_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i18_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i18_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i16_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i16_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i14_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i14_out_ap_vld;
wire   [22:0] grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i12_out;
wire    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i12_out_ap_vld;
wire    grp_quantize_v_group_fu_568_ap_start;
wire    grp_quantize_v_group_fu_568_ap_done;
wire    grp_quantize_v_group_fu_568_ap_idle;
wire    grp_quantize_v_group_fu_568_ap_ready;
wire    grp_quantize_v_group_fu_568_vq_cache_o_stream_TREADY;
wire    grp_quantize_v_group_fu_568_vs_cache_o_stream_TREADY;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_0_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_0_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_0_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_0_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_1_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_1_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_1_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_1_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_2_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_2_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_2_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_2_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_3_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_3_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_3_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_3_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_4_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_4_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_4_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_4_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_5_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_5_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_5_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_5_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_6_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_6_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_6_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_6_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vq_buf_7_address1;
wire    grp_quantize_v_group_fu_568_vq_buf_7_ce1;
wire   [7:0] grp_quantize_v_group_fu_568_vq_buf_7_we1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_buf_7_d1;
wire   [9:0] grp_quantize_v_group_fu_568_vs_buf_address0;
wire    grp_quantize_v_group_fu_568_vs_buf_ce0;
wire   [9:0] grp_quantize_v_group_fu_568_vs_buf_address1;
wire    grp_quantize_v_group_fu_568_vs_buf_ce1;
wire    grp_quantize_v_group_fu_568_vs_buf_we1;
wire   [31:0] grp_quantize_v_group_fu_568_vs_buf_d1;
wire   [63:0] grp_quantize_v_group_fu_568_vq_cache_o_stream_TDATA;
wire    grp_quantize_v_group_fu_568_vq_cache_o_stream_TVALID;
wire   [7:0] grp_quantize_v_group_fu_568_vs_cache_o_stream_TDATA;
wire    grp_quantize_v_group_fu_568_vs_cache_o_stream_TVALID;
wire    grp_shared_rv_bmm_head_fu_653_ap_start;
wire    grp_shared_rv_bmm_head_fu_653_ap_done;
wire    grp_shared_rv_bmm_head_fu_653_ap_idle;
wire    grp_shared_rv_bmm_head_fu_653_ap_ready;
reg   [10:0] grp_shared_rv_bmm_head_fu_653_num_tokens;
reg   [7:0] grp_shared_rv_bmm_head_fu_653_valid_st;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_0_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_0_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_1_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_1_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_2_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_2_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_3_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_3_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_4_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_4_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_5_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_5_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_6_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_6_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vq_buf_7_address0;
wire    grp_shared_rv_bmm_head_fu_653_vq_buf_7_ce0;
wire   [9:0] grp_shared_rv_bmm_head_fu_653_vs_buf_address0;
wire    grp_shared_rv_bmm_head_fu_653_vs_buf_ce0;
wire    grp_shared_rv_bmm_head_fu_653_rq_stream_TREADY;
wire    grp_shared_rv_bmm_head_fu_653_rs_stream_TREADY;
wire   [63:0] grp_shared_rv_bmm_head_fu_653_aq_stream_TDATA;
wire    grp_shared_rv_bmm_head_fu_653_aq_stream_TVALID;
wire    grp_shared_rv_bmm_head_fu_653_aq_stream_TREADY;
wire   [7:0] grp_shared_rv_bmm_head_fu_653_as_stream_TDATA;
wire    grp_shared_rv_bmm_head_fu_653_as_stream_TVALID;
wire    grp_shared_rv_bmm_head_fu_653_as_stream_TREADY;
wire    grp_load_quantize_vit_v_fu_679_ap_start;
wire    grp_load_quantize_vit_v_fu_679_ap_done;
wire    grp_load_quantize_vit_v_fu_679_ap_idle;
wire    grp_load_quantize_vit_v_fu_679_ap_ready;
wire    grp_load_quantize_vit_v_fu_679_v_stream_TREADY;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_0_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_0_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_0_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_0_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_1_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_1_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_1_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_1_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_2_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_2_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_2_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_2_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_3_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_3_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_3_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_3_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_4_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_4_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_4_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_4_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_5_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_5_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_5_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_5_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_6_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_6_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_6_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_6_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vq_buf_7_address1;
wire    grp_load_quantize_vit_v_fu_679_vq_buf_7_ce1;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vq_buf_7_we1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_buf_7_d1;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vs_buf_address0;
wire    grp_load_quantize_vit_v_fu_679_vs_buf_ce0;
wire   [9:0] grp_load_quantize_vit_v_fu_679_vs_buf_address1;
wire    grp_load_quantize_vit_v_fu_679_vs_buf_ce1;
wire    grp_load_quantize_vit_v_fu_679_vs_buf_we1;
wire   [31:0] grp_load_quantize_vit_v_fu_679_vs_buf_d1;
wire   [63:0] grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TDATA;
wire    grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TVALID;
wire    grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TREADY;
wire   [7:0] grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TDATA;
wire    grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TVALID;
wire    grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TREADY;
reg   [3:0] hct_reg_458;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state10;
reg   [1:0] r_reg_469;
wire    ap_CS_fsm_state12;
wire   [0:0] icmp_ln194_fu_868_p2;
reg    grp_load_llm_v_cache_qs_fu_480_ap_start_reg;
wire   [0:0] icmp_ln410_fu_856_p2;
reg    grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start_reg;
wire    ap_CS_fsm_state8;
reg    grp_quantize_v_group_fu_568_ap_start_reg;
wire    ap_CS_fsm_state9;
reg    grp_shared_rv_bmm_head_fu_653_ap_start_reg;
wire   [0:0] icmp_ln414_fu_1140_p2;
wire    ap_CS_fsm_state15;
wire    ap_CS_fsm_state16;
reg    grp_load_quantize_vit_v_fu_679_ap_start_reg;
wire    ap_CS_fsm_state13;
wire   [0:0] icmp_ln404_fu_1159_p2;
wire    ap_CS_fsm_state14;
reg   [2:0] kv_fu_438;
reg   [3:0] h_fu_442;
wire   [3:0] h_2_fu_1165_p2;
wire   [12:0] empty_fu_698_p1;
wire   [12:0] sub_ln78_fu_710_p2;
wire   [9:0] sub_ln78_1_fu_726_p2;
wire   [9:0] trunc_ln78_2_fu_731_p4;
wire   [9:0] decode_pos_chunk_fu_740_p3;
wire   [30:0] trunc_ln86_fu_755_p1;
wire   [10:0] trunc_ln86_1_fu_780_p1;
wire   [10:0] sub_ln102_fu_784_p2;
wire   [31:0] zext_ln86_fu_800_p1;
wire   [31:0] add_ln102_fu_808_p2;
wire   [0:0] tmp_1_fu_814_p3;
wire   [7:0] sub_ln102_1_fu_822_p2;
wire   [7:0] trunc_ln102_3_fu_827_p4;
wire   [0:0] icmp_ln90_fu_803_p2;
wire   [7:0] select_ln102_fu_837_p3;
reg   [15:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
reg    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
reg    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
reg    ap_ST_fsm_state10_blk;
wire    ap_ST_fsm_state11_blk;
reg    ap_ST_fsm_state12_blk;
wire    ap_ST_fsm_state13_blk;
reg    ap_ST_fsm_state14_blk;
wire    ap_ST_fsm_state15_blk;
reg    ap_ST_fsm_state16_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 16'd1;
//#0 grp_load_llm_v_cache_qs_fu_480_ap_start_reg = 1'b0;
//#0 grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start_reg = 1'b0;
//#0 grp_quantize_v_group_fu_568_ap_start_reg = 1'b0;
//#0 grp_shared_rv_bmm_head_fu_653_ap_start_reg = 1'b0;
//#0 grp_load_quantize_vit_v_fu_679_ap_start_reg = 1'b0;
//#0 kv_fu_438 = 3'd0;
//#0 h_fu_442 = 4'd0;
end

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_0_address0),
    .ce0(vq_buf_ce0),
    .q0(vq_buf_q0),
    .address1(vq_buf_address1),
    .ce1(vq_buf_ce1),
    .we1(vq_buf_we1),
    .d1(vq_buf_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_1_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_1_address0),
    .ce0(vq_buf_1_ce0),
    .q0(vq_buf_1_q0),
    .address1(vq_buf_1_address1),
    .ce1(vq_buf_1_ce1),
    .we1(vq_buf_1_we1),
    .d1(vq_buf_1_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_2_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_2_address0),
    .ce0(vq_buf_2_ce0),
    .q0(vq_buf_2_q0),
    .address1(vq_buf_2_address1),
    .ce1(vq_buf_2_ce1),
    .we1(vq_buf_2_we1),
    .d1(vq_buf_2_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_3_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_3_address0),
    .ce0(vq_buf_3_ce0),
    .q0(vq_buf_3_q0),
    .address1(vq_buf_3_address1),
    .ce1(vq_buf_3_ce1),
    .we1(vq_buf_3_we1),
    .d1(vq_buf_3_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_4_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_4_address0),
    .ce0(vq_buf_4_ce0),
    .q0(vq_buf_4_q0),
    .address1(vq_buf_4_address1),
    .ce1(vq_buf_4_ce1),
    .we1(vq_buf_4_we1),
    .d1(vq_buf_4_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_5_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_5_address0),
    .ce0(vq_buf_5_ce0),
    .q0(vq_buf_5_q0),
    .address1(vq_buf_5_address1),
    .ce1(vq_buf_5_ce1),
    .we1(vq_buf_5_we1),
    .d1(vq_buf_5_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_6_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_6_address0),
    .ce0(vq_buf_6_ce0),
    .q0(vq_buf_6_q0),
    .address1(vq_buf_6_address1),
    .ce1(vq_buf_6_ce1),
    .we1(vq_buf_6_we1),
    .d1(vq_buf_6_d1)
);

RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W #(
    .DataWidth( 64 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vq_buf_7_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_fu_653_vq_buf_7_address0),
    .ce0(vq_buf_7_ce0),
    .q0(vq_buf_7_q0),
    .address1(vq_buf_7_address1),
    .ce1(vq_buf_7_ce1),
    .we1(vq_buf_7_we1),
    .d1(vq_buf_7_d1)
);

RV_GEMM_rv_gemm_one_layer_vs_buf_RAM_2P_BRAM_2R1W #(
    .DataWidth( 32 ),
    .AddressRange( 1024 ),
    .AddressWidth( 10 ))
vs_buf_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(vs_buf_address0),
    .ce0(vs_buf_ce0),
    .q0(vs_buf_q0),
    .address1(vs_buf_address1),
    .ce1(vs_buf_ce1),
    .we1(vs_buf_we1),
    .d1(vs_buf_d1)
);

RV_GEMM_load_llm_v_cache_qs grp_load_llm_v_cache_qs_fu_480(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_load_llm_v_cache_qs_fu_480_ap_start),
    .ap_done(grp_load_llm_v_cache_qs_fu_480_ap_done),
    .ap_idle(grp_load_llm_v_cache_qs_fu_480_ap_idle),
    .ap_ready(grp_load_llm_v_cache_qs_fu_480_ap_ready),
    .vq_cache_i_stream_TDATA(vq_cache_i_stream_TDATA),
    .vq_cache_i_stream_TVALID(vq_cache_i_stream_TVALID),
    .vq_cache_i_stream_TREADY(grp_load_llm_v_cache_qs_fu_480_vq_cache_i_stream_TREADY),
    .vs_cache_i_stream_TDATA(vs_cache_i_stream_TDATA),
    .vs_cache_i_stream_TVALID(vs_cache_i_stream_TVALID),
    .vs_cache_i_stream_TREADY(grp_load_llm_v_cache_qs_fu_480_vs_cache_i_stream_TREADY),
    .q_buf_0_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_0_address1),
    .q_buf_0_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_0_ce1),
    .q_buf_0_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_0_we1),
    .q_buf_0_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_0_d1),
    .q_buf_1_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_1_address1),
    .q_buf_1_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_1_ce1),
    .q_buf_1_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_1_we1),
    .q_buf_1_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_1_d1),
    .q_buf_2_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_2_address1),
    .q_buf_2_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_2_ce1),
    .q_buf_2_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_2_we1),
    .q_buf_2_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_2_d1),
    .q_buf_3_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_3_address1),
    .q_buf_3_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_3_ce1),
    .q_buf_3_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_3_we1),
    .q_buf_3_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_3_d1),
    .q_buf_4_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_4_address1),
    .q_buf_4_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_4_ce1),
    .q_buf_4_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_4_we1),
    .q_buf_4_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_4_d1),
    .q_buf_5_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_5_address1),
    .q_buf_5_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_5_ce1),
    .q_buf_5_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_5_we1),
    .q_buf_5_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_5_d1),
    .q_buf_6_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_6_address1),
    .q_buf_6_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_6_ce1),
    .q_buf_6_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_6_we1),
    .q_buf_6_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_6_d1),
    .q_buf_7_address1(grp_load_llm_v_cache_qs_fu_480_q_buf_7_address1),
    .q_buf_7_ce1(grp_load_llm_v_cache_qs_fu_480_q_buf_7_ce1),
    .q_buf_7_we1(grp_load_llm_v_cache_qs_fu_480_q_buf_7_we1),
    .q_buf_7_d1(grp_load_llm_v_cache_qs_fu_480_q_buf_7_d1),
    .s_buf_address0(grp_load_llm_v_cache_qs_fu_480_s_buf_address0),
    .s_buf_ce0(grp_load_llm_v_cache_qs_fu_480_s_buf_ce0),
    .s_buf_q0(vs_buf_q0),
    .s_buf_address1(grp_load_llm_v_cache_qs_fu_480_s_buf_address1),
    .s_buf_ce1(grp_load_llm_v_cache_qs_fu_480_s_buf_ce1),
    .s_buf_we1(grp_load_llm_v_cache_qs_fu_480_s_buf_we1),
    .s_buf_d1(grp_load_llm_v_cache_qs_fu_480_s_buf_d1),
    .valid_st(valid_st_reg_1616)
);

RV_GEMM_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2 grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start),
    .ap_done(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_done),
    .ap_idle(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_idle),
    .ap_ready(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_ready),
    .v_stream_TVALID(v_stream_TVALID),
    .v_stream_TDATA(v_stream_TDATA),
    .v_stream_TREADY(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_v_stream_TREADY),
    .p_0_0_7_0_0_0_i138_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i138_out),
    .p_0_0_7_0_0_0_i138_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i138_out_ap_vld),
    .p_0_0_6_0_0_0_i136_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i136_out),
    .p_0_0_6_0_0_0_i136_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i136_out_ap_vld),
    .p_0_0_5_0_0_0_i134_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i134_out),
    .p_0_0_5_0_0_0_i134_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i134_out_ap_vld),
    .p_0_0_4_0_0_0_i132_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i132_out),
    .p_0_0_4_0_0_0_i132_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i132_out_ap_vld),
    .p_0_0_3_0_0_0_i130_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i130_out),
    .p_0_0_3_0_0_0_i130_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i130_out_ap_vld),
    .p_0_0_2_0_0_0_i128_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i128_out),
    .p_0_0_2_0_0_0_i128_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i128_out_ap_vld),
    .p_0_0_1_0_0_0_i126_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i126_out),
    .p_0_0_1_0_0_0_i126_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i126_out_ap_vld),
    .p_0_0_0_0_0_0_i124_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i124_out),
    .p_0_0_0_0_0_0_i124_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i124_out_ap_vld),
    .p_0_0_7_0_0_0_i122_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i122_out),
    .p_0_0_7_0_0_0_i122_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i122_out_ap_vld),
    .p_0_0_6_0_0_0_i120_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i120_out),
    .p_0_0_6_0_0_0_i120_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i120_out_ap_vld),
    .p_0_0_5_0_0_0_i118_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i118_out),
    .p_0_0_5_0_0_0_i118_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i118_out_ap_vld),
    .p_0_0_4_0_0_0_i116_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i116_out),
    .p_0_0_4_0_0_0_i116_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i116_out_ap_vld),
    .p_0_0_3_0_0_0_i114_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i114_out),
    .p_0_0_3_0_0_0_i114_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i114_out_ap_vld),
    .p_0_0_2_0_0_0_i112_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i112_out),
    .p_0_0_2_0_0_0_i112_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i112_out_ap_vld),
    .p_0_0_1_0_0_0_i110_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i110_out),
    .p_0_0_1_0_0_0_i110_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i110_out_ap_vld),
    .p_0_0_0_0_0_0_i108_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i108_out),
    .p_0_0_0_0_0_0_i108_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i108_out_ap_vld),
    .p_0_0_7_0_0_0_i106_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i106_out),
    .p_0_0_7_0_0_0_i106_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i106_out_ap_vld),
    .p_0_0_6_0_0_0_i104_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i104_out),
    .p_0_0_6_0_0_0_i104_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i104_out_ap_vld),
    .p_0_0_5_0_0_0_i102_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i102_out),
    .p_0_0_5_0_0_0_i102_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i102_out_ap_vld),
    .p_0_0_4_0_0_0_i100_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i100_out),
    .p_0_0_4_0_0_0_i100_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i100_out_ap_vld),
    .p_0_0_3_0_0_0_i98_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i98_out),
    .p_0_0_3_0_0_0_i98_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i98_out_ap_vld),
    .p_0_0_2_0_0_0_i96_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i96_out),
    .p_0_0_2_0_0_0_i96_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i96_out_ap_vld),
    .p_0_0_1_0_0_0_i94_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i94_out),
    .p_0_0_1_0_0_0_i94_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i94_out_ap_vld),
    .p_0_0_0_0_0_0_i92_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i92_out),
    .p_0_0_0_0_0_0_i92_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i92_out_ap_vld),
    .p_0_0_7_0_0_0_i90_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i90_out),
    .p_0_0_7_0_0_0_i90_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i90_out_ap_vld),
    .p_0_0_6_0_0_0_i88_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i88_out),
    .p_0_0_6_0_0_0_i88_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i88_out_ap_vld),
    .p_0_0_5_0_0_0_i86_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i86_out),
    .p_0_0_5_0_0_0_i86_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i86_out_ap_vld),
    .p_0_0_4_0_0_0_i84_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i84_out),
    .p_0_0_4_0_0_0_i84_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i84_out_ap_vld),
    .p_0_0_3_0_0_0_i82_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i82_out),
    .p_0_0_3_0_0_0_i82_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i82_out_ap_vld),
    .p_0_0_2_0_0_0_i80_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i80_out),
    .p_0_0_2_0_0_0_i80_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i80_out_ap_vld),
    .p_0_0_1_0_0_0_i78_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i78_out),
    .p_0_0_1_0_0_0_i78_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i78_out_ap_vld),
    .p_0_0_0_0_0_0_i76_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i76_out),
    .p_0_0_0_0_0_0_i76_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i76_out_ap_vld),
    .p_0_0_7_0_0_0_i74_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i74_out),
    .p_0_0_7_0_0_0_i74_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i74_out_ap_vld),
    .p_0_0_6_0_0_0_i72_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i72_out),
    .p_0_0_6_0_0_0_i72_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i72_out_ap_vld),
    .p_0_0_5_0_0_0_i70_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i70_out),
    .p_0_0_5_0_0_0_i70_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i70_out_ap_vld),
    .p_0_0_4_0_0_0_i68_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i68_out),
    .p_0_0_4_0_0_0_i68_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i68_out_ap_vld),
    .p_0_0_3_0_0_0_i66_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i66_out),
    .p_0_0_3_0_0_0_i66_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i66_out_ap_vld),
    .p_0_0_2_0_0_0_i64_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i64_out),
    .p_0_0_2_0_0_0_i64_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i64_out_ap_vld),
    .p_0_0_1_0_0_0_i62_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i62_out),
    .p_0_0_1_0_0_0_i62_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i62_out_ap_vld),
    .p_0_0_0_0_0_0_i60_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i60_out),
    .p_0_0_0_0_0_0_i60_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i60_out_ap_vld),
    .p_0_0_7_0_0_0_i58_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i58_out),
    .p_0_0_7_0_0_0_i58_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i58_out_ap_vld),
    .p_0_0_6_0_0_0_i56_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i56_out),
    .p_0_0_6_0_0_0_i56_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i56_out_ap_vld),
    .p_0_0_5_0_0_0_i54_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i54_out),
    .p_0_0_5_0_0_0_i54_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i54_out_ap_vld),
    .p_0_0_4_0_0_0_i52_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i52_out),
    .p_0_0_4_0_0_0_i52_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i52_out_ap_vld),
    .p_0_0_3_0_0_0_i50_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i50_out),
    .p_0_0_3_0_0_0_i50_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i50_out_ap_vld),
    .p_0_0_2_0_0_0_i48_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i48_out),
    .p_0_0_2_0_0_0_i48_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i48_out_ap_vld),
    .p_0_0_1_0_0_0_i46_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i46_out),
    .p_0_0_1_0_0_0_i46_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i46_out_ap_vld),
    .p_0_0_0_0_0_0_i44_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i44_out),
    .p_0_0_0_0_0_0_i44_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i44_out_ap_vld),
    .p_0_0_7_0_0_0_i42_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i42_out),
    .p_0_0_7_0_0_0_i42_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i42_out_ap_vld),
    .p_0_0_6_0_0_0_i40_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i40_out),
    .p_0_0_6_0_0_0_i40_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i40_out_ap_vld),
    .p_0_0_5_0_0_0_i38_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i38_out),
    .p_0_0_5_0_0_0_i38_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i38_out_ap_vld),
    .p_0_0_4_0_0_0_i36_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i36_out),
    .p_0_0_4_0_0_0_i36_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i36_out_ap_vld),
    .p_0_0_3_0_0_0_i34_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i34_out),
    .p_0_0_3_0_0_0_i34_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i34_out_ap_vld),
    .p_0_0_2_0_0_0_i32_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i32_out),
    .p_0_0_2_0_0_0_i32_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i32_out_ap_vld),
    .p_0_0_1_0_0_0_i30_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i30_out),
    .p_0_0_1_0_0_0_i30_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i30_out_ap_vld),
    .p_0_0_0_0_0_0_i28_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i28_out),
    .p_0_0_0_0_0_0_i28_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i28_out_ap_vld),
    .p_0_0_7_0_0_0_i26_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i26_out),
    .p_0_0_7_0_0_0_i26_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i26_out_ap_vld),
    .p_0_0_6_0_0_0_i24_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i24_out),
    .p_0_0_6_0_0_0_i24_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i24_out_ap_vld),
    .p_0_0_5_0_0_0_i22_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i22_out),
    .p_0_0_5_0_0_0_i22_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i22_out_ap_vld),
    .p_0_0_4_0_0_0_i20_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i20_out),
    .p_0_0_4_0_0_0_i20_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i20_out_ap_vld),
    .p_0_0_3_0_0_0_i18_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i18_out),
    .p_0_0_3_0_0_0_i18_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i18_out_ap_vld),
    .p_0_0_2_0_0_0_i16_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i16_out),
    .p_0_0_2_0_0_0_i16_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i16_out_ap_vld),
    .p_0_0_1_0_0_0_i14_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i14_out),
    .p_0_0_1_0_0_0_i14_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i14_out_ap_vld),
    .p_0_0_0_0_0_0_i12_out(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i12_out),
    .p_0_0_0_0_0_0_i12_out_ap_vld(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i12_out_ap_vld)
);

RV_GEMM_quantize_v_group grp_quantize_v_group_fu_568(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_quantize_v_group_fu_568_ap_start),
    .ap_done(grp_quantize_v_group_fu_568_ap_done),
    .ap_idle(grp_quantize_v_group_fu_568_ap_idle),
    .ap_ready(grp_quantize_v_group_fu_568_ap_ready),
    .vq_cache_o_stream_TREADY(grp_quantize_v_group_fu_568_vq_cache_o_stream_TREADY),
    .vs_cache_o_stream_TREADY(grp_quantize_v_group_fu_568_vs_cache_o_stream_TREADY),
    .hct(trunc_ln194_reg_1638),
    .st(decoded_chunk_reg_1581),
    .raw_tile_0_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i108_out),
    .raw_tile_0_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i92_out),
    .raw_tile_0_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i76_out),
    .raw_tile_0_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i60_out),
    .raw_tile_0_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i44_out),
    .raw_tile_0_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i28_out),
    .raw_tile_0_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i12_out),
    .raw_tile_0_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_0_0_0_0_i124_out),
    .raw_tile_1_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i110_out),
    .raw_tile_1_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i94_out),
    .raw_tile_1_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i78_out),
    .raw_tile_1_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i62_out),
    .raw_tile_1_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i46_out),
    .raw_tile_1_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i30_out),
    .raw_tile_1_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i14_out),
    .raw_tile_1_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_1_0_0_0_i126_out),
    .raw_tile_2_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i112_out),
    .raw_tile_2_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i96_out),
    .raw_tile_2_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i80_out),
    .raw_tile_2_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i64_out),
    .raw_tile_2_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i48_out),
    .raw_tile_2_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i32_out),
    .raw_tile_2_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i16_out),
    .raw_tile_2_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_2_0_0_0_i128_out),
    .raw_tile_3_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i114_out),
    .raw_tile_3_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i98_out),
    .raw_tile_3_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i82_out),
    .raw_tile_3_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i66_out),
    .raw_tile_3_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i50_out),
    .raw_tile_3_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i34_out),
    .raw_tile_3_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i18_out),
    .raw_tile_3_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_3_0_0_0_i130_out),
    .raw_tile_4_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i116_out),
    .raw_tile_4_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i100_out),
    .raw_tile_4_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i84_out),
    .raw_tile_4_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i68_out),
    .raw_tile_4_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i52_out),
    .raw_tile_4_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i36_out),
    .raw_tile_4_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i20_out),
    .raw_tile_4_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_4_0_0_0_i132_out),
    .raw_tile_5_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i118_out),
    .raw_tile_5_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i102_out),
    .raw_tile_5_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i86_out),
    .raw_tile_5_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i70_out),
    .raw_tile_5_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i54_out),
    .raw_tile_5_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i38_out),
    .raw_tile_5_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i22_out),
    .raw_tile_5_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_5_0_0_0_i134_out),
    .raw_tile_6_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i120_out),
    .raw_tile_6_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i104_out),
    .raw_tile_6_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i88_out),
    .raw_tile_6_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i72_out),
    .raw_tile_6_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i56_out),
    .raw_tile_6_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i40_out),
    .raw_tile_6_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i24_out),
    .raw_tile_6_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_6_0_0_0_i136_out),
    .raw_tile_7_0_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i122_out),
    .raw_tile_7_1_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i106_out),
    .raw_tile_7_2_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i90_out),
    .raw_tile_7_3_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i74_out),
    .raw_tile_7_4_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i58_out),
    .raw_tile_7_5_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i42_out),
    .raw_tile_7_6_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i26_out),
    .raw_tile_7_7_val(grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_p_0_0_7_0_0_0_i138_out),
    .write_cache_stream(1'd1),
    .vq_buf_0_address1(grp_quantize_v_group_fu_568_vq_buf_0_address1),
    .vq_buf_0_ce1(grp_quantize_v_group_fu_568_vq_buf_0_ce1),
    .vq_buf_0_we1(grp_quantize_v_group_fu_568_vq_buf_0_we1),
    .vq_buf_0_d1(grp_quantize_v_group_fu_568_vq_buf_0_d1),
    .vq_buf_1_address1(grp_quantize_v_group_fu_568_vq_buf_1_address1),
    .vq_buf_1_ce1(grp_quantize_v_group_fu_568_vq_buf_1_ce1),
    .vq_buf_1_we1(grp_quantize_v_group_fu_568_vq_buf_1_we1),
    .vq_buf_1_d1(grp_quantize_v_group_fu_568_vq_buf_1_d1),
    .vq_buf_2_address1(grp_quantize_v_group_fu_568_vq_buf_2_address1),
    .vq_buf_2_ce1(grp_quantize_v_group_fu_568_vq_buf_2_ce1),
    .vq_buf_2_we1(grp_quantize_v_group_fu_568_vq_buf_2_we1),
    .vq_buf_2_d1(grp_quantize_v_group_fu_568_vq_buf_2_d1),
    .vq_buf_3_address1(grp_quantize_v_group_fu_568_vq_buf_3_address1),
    .vq_buf_3_ce1(grp_quantize_v_group_fu_568_vq_buf_3_ce1),
    .vq_buf_3_we1(grp_quantize_v_group_fu_568_vq_buf_3_we1),
    .vq_buf_3_d1(grp_quantize_v_group_fu_568_vq_buf_3_d1),
    .vq_buf_4_address1(grp_quantize_v_group_fu_568_vq_buf_4_address1),
    .vq_buf_4_ce1(grp_quantize_v_group_fu_568_vq_buf_4_ce1),
    .vq_buf_4_we1(grp_quantize_v_group_fu_568_vq_buf_4_we1),
    .vq_buf_4_d1(grp_quantize_v_group_fu_568_vq_buf_4_d1),
    .vq_buf_5_address1(grp_quantize_v_group_fu_568_vq_buf_5_address1),
    .vq_buf_5_ce1(grp_quantize_v_group_fu_568_vq_buf_5_ce1),
    .vq_buf_5_we1(grp_quantize_v_group_fu_568_vq_buf_5_we1),
    .vq_buf_5_d1(grp_quantize_v_group_fu_568_vq_buf_5_d1),
    .vq_buf_6_address1(grp_quantize_v_group_fu_568_vq_buf_6_address1),
    .vq_buf_6_ce1(grp_quantize_v_group_fu_568_vq_buf_6_ce1),
    .vq_buf_6_we1(grp_quantize_v_group_fu_568_vq_buf_6_we1),
    .vq_buf_6_d1(grp_quantize_v_group_fu_568_vq_buf_6_d1),
    .vq_buf_7_address1(grp_quantize_v_group_fu_568_vq_buf_7_address1),
    .vq_buf_7_ce1(grp_quantize_v_group_fu_568_vq_buf_7_ce1),
    .vq_buf_7_we1(grp_quantize_v_group_fu_568_vq_buf_7_we1),
    .vq_buf_7_d1(grp_quantize_v_group_fu_568_vq_buf_7_d1),
    .vs_buf_address0(grp_quantize_v_group_fu_568_vs_buf_address0),
    .vs_buf_ce0(grp_quantize_v_group_fu_568_vs_buf_ce0),
    .vs_buf_q0(vs_buf_q0),
    .vs_buf_address1(grp_quantize_v_group_fu_568_vs_buf_address1),
    .vs_buf_ce1(grp_quantize_v_group_fu_568_vs_buf_ce1),
    .vs_buf_we1(grp_quantize_v_group_fu_568_vs_buf_we1),
    .vs_buf_d1(grp_quantize_v_group_fu_568_vs_buf_d1),
    .vq_cache_o_stream_TDATA(grp_quantize_v_group_fu_568_vq_cache_o_stream_TDATA),
    .vq_cache_o_stream_TVALID(grp_quantize_v_group_fu_568_vq_cache_o_stream_TVALID),
    .vs_cache_o_stream_TDATA(grp_quantize_v_group_fu_568_vs_cache_o_stream_TDATA),
    .vs_cache_o_stream_TVALID(grp_quantize_v_group_fu_568_vs_cache_o_stream_TVALID)
);

RV_GEMM_shared_rv_bmm_head grp_shared_rv_bmm_head_fu_653(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_shared_rv_bmm_head_fu_653_ap_start),
    .ap_done(grp_shared_rv_bmm_head_fu_653_ap_done),
    .ap_idle(grp_shared_rv_bmm_head_fu_653_ap_idle),
    .ap_ready(grp_shared_rv_bmm_head_fu_653_ap_ready),
    .num_tokens(grp_shared_rv_bmm_head_fu_653_num_tokens),
    .valid_st(grp_shared_rv_bmm_head_fu_653_valid_st),
    .vq_buf_0_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_0_address0),
    .vq_buf_0_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_0_ce0),
    .vq_buf_0_q0(vq_buf_q0),
    .vq_buf_1_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_1_address0),
    .vq_buf_1_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_1_ce0),
    .vq_buf_1_q0(vq_buf_1_q0),
    .vq_buf_2_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_2_address0),
    .vq_buf_2_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_2_ce0),
    .vq_buf_2_q0(vq_buf_2_q0),
    .vq_buf_3_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_3_address0),
    .vq_buf_3_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_3_ce0),
    .vq_buf_3_q0(vq_buf_3_q0),
    .vq_buf_4_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_4_address0),
    .vq_buf_4_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_4_ce0),
    .vq_buf_4_q0(vq_buf_4_q0),
    .vq_buf_5_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_5_address0),
    .vq_buf_5_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_5_ce0),
    .vq_buf_5_q0(vq_buf_5_q0),
    .vq_buf_6_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_6_address0),
    .vq_buf_6_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_6_ce0),
    .vq_buf_6_q0(vq_buf_6_q0),
    .vq_buf_7_address0(grp_shared_rv_bmm_head_fu_653_vq_buf_7_address0),
    .vq_buf_7_ce0(grp_shared_rv_bmm_head_fu_653_vq_buf_7_ce0),
    .vq_buf_7_q0(vq_buf_7_q0),
    .vs_buf_address0(grp_shared_rv_bmm_head_fu_653_vs_buf_address0),
    .vs_buf_ce0(grp_shared_rv_bmm_head_fu_653_vs_buf_ce0),
    .vs_buf_q0(vs_buf_q0),
    .rq_stream_TDATA(rq_stream_TDATA),
    .rq_stream_TVALID(rq_stream_TVALID),
    .rq_stream_TREADY(grp_shared_rv_bmm_head_fu_653_rq_stream_TREADY),
    .rs_stream_TDATA(rs_stream_TDATA),
    .rs_stream_TVALID(rs_stream_TVALID),
    .rs_stream_TREADY(grp_shared_rv_bmm_head_fu_653_rs_stream_TREADY),
    .aq_stream_TDATA(grp_shared_rv_bmm_head_fu_653_aq_stream_TDATA),
    .aq_stream_TVALID(grp_shared_rv_bmm_head_fu_653_aq_stream_TVALID),
    .aq_stream_TREADY(grp_shared_rv_bmm_head_fu_653_aq_stream_TREADY),
    .as_stream_TDATA(grp_shared_rv_bmm_head_fu_653_as_stream_TDATA),
    .as_stream_TVALID(grp_shared_rv_bmm_head_fu_653_as_stream_TVALID),
    .as_stream_TREADY(grp_shared_rv_bmm_head_fu_653_as_stream_TREADY)
);

RV_GEMM_load_quantize_vit_v grp_load_quantize_vit_v_fu_679(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_load_quantize_vit_v_fu_679_ap_start),
    .ap_done(grp_load_quantize_vit_v_fu_679_ap_done),
    .ap_idle(grp_load_quantize_vit_v_fu_679_ap_idle),
    .ap_ready(grp_load_quantize_vit_v_fu_679_ap_ready),
    .v_stream_TDATA(v_stream_TDATA),
    .v_stream_TVALID(v_stream_TVALID),
    .v_stream_TREADY(grp_load_quantize_vit_v_fu_679_v_stream_TREADY),
    .vq_buf_0_address1(grp_load_quantize_vit_v_fu_679_vq_buf_0_address1),
    .vq_buf_0_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_0_ce1),
    .vq_buf_0_we1(grp_load_quantize_vit_v_fu_679_vq_buf_0_we1),
    .vq_buf_0_d1(grp_load_quantize_vit_v_fu_679_vq_buf_0_d1),
    .vq_buf_1_address1(grp_load_quantize_vit_v_fu_679_vq_buf_1_address1),
    .vq_buf_1_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_1_ce1),
    .vq_buf_1_we1(grp_load_quantize_vit_v_fu_679_vq_buf_1_we1),
    .vq_buf_1_d1(grp_load_quantize_vit_v_fu_679_vq_buf_1_d1),
    .vq_buf_2_address1(grp_load_quantize_vit_v_fu_679_vq_buf_2_address1),
    .vq_buf_2_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_2_ce1),
    .vq_buf_2_we1(grp_load_quantize_vit_v_fu_679_vq_buf_2_we1),
    .vq_buf_2_d1(grp_load_quantize_vit_v_fu_679_vq_buf_2_d1),
    .vq_buf_3_address1(grp_load_quantize_vit_v_fu_679_vq_buf_3_address1),
    .vq_buf_3_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_3_ce1),
    .vq_buf_3_we1(grp_load_quantize_vit_v_fu_679_vq_buf_3_we1),
    .vq_buf_3_d1(grp_load_quantize_vit_v_fu_679_vq_buf_3_d1),
    .vq_buf_4_address1(grp_load_quantize_vit_v_fu_679_vq_buf_4_address1),
    .vq_buf_4_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_4_ce1),
    .vq_buf_4_we1(grp_load_quantize_vit_v_fu_679_vq_buf_4_we1),
    .vq_buf_4_d1(grp_load_quantize_vit_v_fu_679_vq_buf_4_d1),
    .vq_buf_5_address1(grp_load_quantize_vit_v_fu_679_vq_buf_5_address1),
    .vq_buf_5_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_5_ce1),
    .vq_buf_5_we1(grp_load_quantize_vit_v_fu_679_vq_buf_5_we1),
    .vq_buf_5_d1(grp_load_quantize_vit_v_fu_679_vq_buf_5_d1),
    .vq_buf_6_address1(grp_load_quantize_vit_v_fu_679_vq_buf_6_address1),
    .vq_buf_6_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_6_ce1),
    .vq_buf_6_we1(grp_load_quantize_vit_v_fu_679_vq_buf_6_we1),
    .vq_buf_6_d1(grp_load_quantize_vit_v_fu_679_vq_buf_6_d1),
    .vq_buf_7_address1(grp_load_quantize_vit_v_fu_679_vq_buf_7_address1),
    .vq_buf_7_ce1(grp_load_quantize_vit_v_fu_679_vq_buf_7_ce1),
    .vq_buf_7_we1(grp_load_quantize_vit_v_fu_679_vq_buf_7_we1),
    .vq_buf_7_d1(grp_load_quantize_vit_v_fu_679_vq_buf_7_d1),
    .vs_buf_address0(grp_load_quantize_vit_v_fu_679_vs_buf_address0),
    .vs_buf_ce0(grp_load_quantize_vit_v_fu_679_vs_buf_ce0),
    .vs_buf_q0(vs_buf_q0),
    .vs_buf_address1(grp_load_quantize_vit_v_fu_679_vs_buf_address1),
    .vs_buf_ce1(grp_load_quantize_vit_v_fu_679_vs_buf_ce1),
    .vs_buf_we1(grp_load_quantize_vit_v_fu_679_vs_buf_we1),
    .vs_buf_d1(grp_load_quantize_vit_v_fu_679_vs_buf_d1),
    .vq_cache_o_stream_TDATA(grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TDATA),
    .vq_cache_o_stream_TVALID(grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TVALID),
    .vq_cache_o_stream_TREADY(grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TREADY),
    .vs_cache_o_stream_TDATA(grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TDATA),
    .vs_cache_o_stream_TVALID(grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TVALID),
    .vs_cache_o_stream_TREADY(grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TREADY)
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
        grp_load_llm_v_cache_qs_fu_480_ap_start_reg <= 1'b0;
    end else begin
        if (((mode_val_read_read_fu_452_p2 == 1'd0) & (icmp_ln410_fu_856_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state5))) begin
            grp_load_llm_v_cache_qs_fu_480_ap_start_reg <= 1'b1;
        end else if ((grp_load_llm_v_cache_qs_fu_480_ap_ready == 1'b1)) begin
            grp_load_llm_v_cache_qs_fu_480_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_load_quantize_vit_v_fu_679_ap_start_reg <= 1'b0;
    end else begin
        if (((icmp_ln404_fu_1159_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state13))) begin
            grp_load_quantize_vit_v_fu_679_ap_start_reg <= 1'b1;
        end else if ((grp_load_quantize_vit_v_fu_679_ap_ready == 1'b1)) begin
            grp_load_quantize_vit_v_fu_679_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_quantize_v_group_fu_568_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state9)) begin
            grp_quantize_v_group_fu_568_ap_start_reg <= 1'b1;
        end else if ((grp_quantize_v_group_fu_568_ap_ready == 1'b1)) begin
            grp_quantize_v_group_fu_568_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start_reg <= 1'b0;
    end else begin
        if (((icmp_ln194_fu_868_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state7))) begin
            grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start_reg <= 1'b1;
        end else if ((grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_ready == 1'b1)) begin
            grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_shared_rv_bmm_head_fu_653_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state15) | ((icmp_ln414_fu_1140_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state11)))) begin
            grp_shared_rv_bmm_head_fu_653_ap_start_reg <= 1'b1;
        end else if ((grp_shared_rv_bmm_head_fu_653_ap_ready == 1'b1)) begin
            grp_shared_rv_bmm_head_fu_653_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((mode_val_read_read_fu_452_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        h_fu_442 <= 4'd0;
    end else if (((icmp_ln404_fu_1159_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state13))) begin
        h_fu_442 <= h_2_fu_1165_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((grp_quantize_v_group_fu_568_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        hct_reg_458 <= hct_1_reg_1633;
    end else if (((grp_load_llm_v_cache_qs_fu_480_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state6))) begin
        hct_reg_458 <= 4'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((mode_val_read_read_fu_452_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        kv_fu_438 <= 3'd0;
    end else if (((icmp_ln414_fu_1140_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state11))) begin
        kv_fu_438 <= kv_2_reg_1625;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln194_fu_868_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state7))) begin
        r_reg_469 <= 2'd0;
    end else if (((grp_shared_rv_bmm_head_fu_653_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state12))) begin
        r_reg_469 <= r_1_reg_1838;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        add_ln86_reg_1593 <= add_ln86_fu_758_p2;
        decoded_chunk_reg_1581 <= decoded_chunk_fu_747_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        hct_1_reg_1633 <= hct_1_fu_874_p2;
        trunc_ln194_reg_1638 <= trunc_ln194_fu_880_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        kv_2_reg_1625 <= kv_2_fu_862_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        r_1_reg_1838 <= r_1_fu_1146_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        tmp_reg_1566 <= pos_r[32'd31];
        trunc_ln78_1_reg_1572 <= {{sub_ln78_fu_710_p2[12:3]}};
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        trunc_ln102_2_reg_1611 <= {{sub_ln102_fu_784_p2[10:3]}};
        valid_s_reg_1605 <= valid_s_fu_774_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        valid_st_reg_1616 <= valid_st_fu_845_p3;
    end
end

always @ (*) begin
    if ((grp_quantize_v_group_fu_568_ap_done == 1'b0)) begin
        ap_ST_fsm_state10_blk = 1'b1;
    end else begin
        ap_ST_fsm_state10_blk = 1'b0;
    end
end

assign ap_ST_fsm_state11_blk = 1'b0;

always @ (*) begin
    if ((grp_shared_rv_bmm_head_fu_653_ap_done == 1'b0)) begin
        ap_ST_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_fsm_state12_blk = 1'b0;
    end
end

assign ap_ST_fsm_state13_blk = 1'b0;

always @ (*) begin
    if ((grp_load_quantize_vit_v_fu_679_ap_done == 1'b0)) begin
        ap_ST_fsm_state14_blk = 1'b1;
    end else begin
        ap_ST_fsm_state14_blk = 1'b0;
    end
end

assign ap_ST_fsm_state15_blk = 1'b0;

always @ (*) begin
    if ((grp_shared_rv_bmm_head_fu_653_ap_done == 1'b0)) begin
        ap_ST_fsm_state16_blk = 1'b1;
    end else begin
        ap_ST_fsm_state16_blk = 1'b0;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

always @ (*) begin
    if ((grp_load_llm_v_cache_qs_fu_480_ap_done == 1'b0)) begin
        ap_ST_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_fsm_state6_blk = 1'b0;
    end
end

assign ap_ST_fsm_state7_blk = 1'b0;

always @ (*) begin
    if ((grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_done == 1'b0)) begin
        ap_ST_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_fsm_state8_blk = 1'b0;
    end
end

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if ((((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)) | ((1'b1 == ap_CS_fsm_state5) & ((mode_val_read_read_fu_452_p2 == 1'd1) | (icmp_ln410_fu_856_p2 == 1'd1))))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
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
    if (((1'b1 == ap_CS_fsm_state5) & ((mode_val_read_read_fu_452_p2 == 1'd1) | (icmp_ln410_fu_856_p2 == 1'd1)))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        aq_stream_TVALID = grp_shared_rv_bmm_head_fu_653_aq_stream_TVALID;
    end else begin
        aq_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        as_stream_TVALID = grp_shared_rv_bmm_head_fu_653_as_stream_TVALID;
    end else begin
        as_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state16)) begin
        grp_shared_rv_bmm_head_fu_653_num_tokens = 11'd1024;
    end else if ((1'b1 == ap_CS_fsm_state12)) begin
        grp_shared_rv_bmm_head_fu_653_num_tokens = 11'd8;
    end else begin
        grp_shared_rv_bmm_head_fu_653_num_tokens = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state16)) begin
        grp_shared_rv_bmm_head_fu_653_valid_st = 8'd128;
    end else if ((1'b1 == ap_CS_fsm_state12)) begin
        grp_shared_rv_bmm_head_fu_653_valid_st = valid_st_reg_1616;
    end else begin
        grp_shared_rv_bmm_head_fu_653_valid_st = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        rq_stream_TREADY = grp_shared_rv_bmm_head_fu_653_rq_stream_TREADY;
    end else begin
        rq_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        rs_stream_TREADY = grp_shared_rv_bmm_head_fu_653_rs_stream_TREADY;
    end else begin
        rs_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        v_stream_TREADY = grp_load_quantize_vit_v_fu_679_v_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state8)) begin
        v_stream_TREADY = grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_v_stream_TREADY;
    end else begin
        v_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_1_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_1_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_1_address1 = grp_quantize_v_group_fu_568_vq_buf_1_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_1_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_1_address1;
    end else begin
        vq_buf_1_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_1_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_1_ce0;
    end else begin
        vq_buf_1_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_1_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_1_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_1_ce1 = grp_quantize_v_group_fu_568_vq_buf_1_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_1_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_1_ce1;
    end else begin
        vq_buf_1_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_1_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_1_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_1_d1 = grp_quantize_v_group_fu_568_vq_buf_1_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_1_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_1_d1;
    end else begin
        vq_buf_1_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_1_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_1_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_1_we1 = grp_quantize_v_group_fu_568_vq_buf_1_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_1_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_1_we1;
    end else begin
        vq_buf_1_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_2_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_2_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_2_address1 = grp_quantize_v_group_fu_568_vq_buf_2_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_2_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_2_address1;
    end else begin
        vq_buf_2_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_2_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_2_ce0;
    end else begin
        vq_buf_2_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_2_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_2_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_2_ce1 = grp_quantize_v_group_fu_568_vq_buf_2_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_2_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_2_ce1;
    end else begin
        vq_buf_2_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_2_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_2_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_2_d1 = grp_quantize_v_group_fu_568_vq_buf_2_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_2_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_2_d1;
    end else begin
        vq_buf_2_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_2_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_2_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_2_we1 = grp_quantize_v_group_fu_568_vq_buf_2_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_2_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_2_we1;
    end else begin
        vq_buf_2_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_3_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_3_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_3_address1 = grp_quantize_v_group_fu_568_vq_buf_3_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_3_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_3_address1;
    end else begin
        vq_buf_3_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_3_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_3_ce0;
    end else begin
        vq_buf_3_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_3_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_3_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_3_ce1 = grp_quantize_v_group_fu_568_vq_buf_3_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_3_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_3_ce1;
    end else begin
        vq_buf_3_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_3_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_3_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_3_d1 = grp_quantize_v_group_fu_568_vq_buf_3_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_3_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_3_d1;
    end else begin
        vq_buf_3_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_3_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_3_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_3_we1 = grp_quantize_v_group_fu_568_vq_buf_3_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_3_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_3_we1;
    end else begin
        vq_buf_3_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_4_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_4_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_4_address1 = grp_quantize_v_group_fu_568_vq_buf_4_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_4_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_4_address1;
    end else begin
        vq_buf_4_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_4_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_4_ce0;
    end else begin
        vq_buf_4_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_4_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_4_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_4_ce1 = grp_quantize_v_group_fu_568_vq_buf_4_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_4_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_4_ce1;
    end else begin
        vq_buf_4_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_4_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_4_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_4_d1 = grp_quantize_v_group_fu_568_vq_buf_4_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_4_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_4_d1;
    end else begin
        vq_buf_4_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_4_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_4_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_4_we1 = grp_quantize_v_group_fu_568_vq_buf_4_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_4_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_4_we1;
    end else begin
        vq_buf_4_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_5_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_5_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_5_address1 = grp_quantize_v_group_fu_568_vq_buf_5_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_5_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_5_address1;
    end else begin
        vq_buf_5_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_5_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_5_ce0;
    end else begin
        vq_buf_5_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_5_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_5_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_5_ce1 = grp_quantize_v_group_fu_568_vq_buf_5_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_5_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_5_ce1;
    end else begin
        vq_buf_5_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_5_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_5_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_5_d1 = grp_quantize_v_group_fu_568_vq_buf_5_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_5_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_5_d1;
    end else begin
        vq_buf_5_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_5_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_5_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_5_we1 = grp_quantize_v_group_fu_568_vq_buf_5_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_5_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_5_we1;
    end else begin
        vq_buf_5_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_6_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_6_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_6_address1 = grp_quantize_v_group_fu_568_vq_buf_6_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_6_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_6_address1;
    end else begin
        vq_buf_6_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_6_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_6_ce0;
    end else begin
        vq_buf_6_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_6_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_6_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_6_ce1 = grp_quantize_v_group_fu_568_vq_buf_6_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_6_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_6_ce1;
    end else begin
        vq_buf_6_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_6_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_6_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_6_d1 = grp_quantize_v_group_fu_568_vq_buf_6_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_6_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_6_d1;
    end else begin
        vq_buf_6_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_6_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_6_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_6_we1 = grp_quantize_v_group_fu_568_vq_buf_6_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_6_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_6_we1;
    end else begin
        vq_buf_6_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_7_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_7_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_7_address1 = grp_quantize_v_group_fu_568_vq_buf_7_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_7_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_7_address1;
    end else begin
        vq_buf_7_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_7_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_7_ce0;
    end else begin
        vq_buf_7_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_7_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_7_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_7_ce1 = grp_quantize_v_group_fu_568_vq_buf_7_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_7_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_7_ce1;
    end else begin
        vq_buf_7_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_7_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_7_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_7_d1 = grp_quantize_v_group_fu_568_vq_buf_7_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_7_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_7_d1;
    end else begin
        vq_buf_7_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_7_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_7_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_7_we1 = grp_quantize_v_group_fu_568_vq_buf_7_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_7_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_7_we1;
    end else begin
        vq_buf_7_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_address1 = grp_load_quantize_vit_v_fu_679_vq_buf_0_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_address1 = grp_quantize_v_group_fu_568_vq_buf_0_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_address1 = grp_load_llm_v_cache_qs_fu_480_q_buf_0_address1;
    end else begin
        vq_buf_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vq_buf_ce0 = grp_shared_rv_bmm_head_fu_653_vq_buf_0_ce0;
    end else begin
        vq_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_ce1 = grp_load_quantize_vit_v_fu_679_vq_buf_0_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_ce1 = grp_quantize_v_group_fu_568_vq_buf_0_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_ce1 = grp_load_llm_v_cache_qs_fu_480_q_buf_0_ce1;
    end else begin
        vq_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_d1 = grp_load_quantize_vit_v_fu_679_vq_buf_0_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_d1 = grp_quantize_v_group_fu_568_vq_buf_0_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_d1 = grp_load_llm_v_cache_qs_fu_480_q_buf_0_d1;
    end else begin
        vq_buf_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_buf_we1 = grp_load_quantize_vit_v_fu_679_vq_buf_0_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_buf_we1 = grp_quantize_v_group_fu_568_vq_buf_0_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_buf_we1 = grp_load_llm_v_cache_qs_fu_480_q_buf_0_we1;
    end else begin
        vq_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        vq_cache_i_stream_TREADY = grp_load_llm_v_cache_qs_fu_480_vq_cache_i_stream_TREADY;
    end else begin
        vq_cache_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
        vq_cache_o_stream_TDATA = grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TDATA;
    end else if (((grp_quantize_v_group_fu_568_vq_cache_o_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        vq_cache_o_stream_TDATA = grp_quantize_v_group_fu_568_vq_cache_o_stream_TDATA;
    end else begin
        vq_cache_o_stream_TDATA = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vq_cache_o_stream_TVALID = grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vq_cache_o_stream_TVALID = grp_quantize_v_group_fu_568_vq_cache_o_stream_TVALID;
    end else begin
        vq_cache_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_buf_address0 = grp_load_quantize_vit_v_fu_679_vs_buf_address0;
    end else if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vs_buf_address0 = grp_shared_rv_bmm_head_fu_653_vs_buf_address0;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_buf_address0 = grp_quantize_v_group_fu_568_vs_buf_address0;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_buf_address0 = grp_load_llm_v_cache_qs_fu_480_s_buf_address0;
    end else begin
        vs_buf_address0 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_buf_address1 = grp_load_quantize_vit_v_fu_679_vs_buf_address1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_buf_address1 = grp_quantize_v_group_fu_568_vs_buf_address1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_buf_address1 = grp_load_llm_v_cache_qs_fu_480_s_buf_address1;
    end else begin
        vs_buf_address1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_buf_ce0 = grp_load_quantize_vit_v_fu_679_vs_buf_ce0;
    end else if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state12))) begin
        vs_buf_ce0 = grp_shared_rv_bmm_head_fu_653_vs_buf_ce0;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_buf_ce0 = grp_quantize_v_group_fu_568_vs_buf_ce0;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_buf_ce0 = grp_load_llm_v_cache_qs_fu_480_s_buf_ce0;
    end else begin
        vs_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_buf_ce1 = grp_load_quantize_vit_v_fu_679_vs_buf_ce1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_buf_ce1 = grp_quantize_v_group_fu_568_vs_buf_ce1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_buf_ce1 = grp_load_llm_v_cache_qs_fu_480_s_buf_ce1;
    end else begin
        vs_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_buf_d1 = grp_load_quantize_vit_v_fu_679_vs_buf_d1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_buf_d1 = grp_quantize_v_group_fu_568_vs_buf_d1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_buf_d1 = grp_load_llm_v_cache_qs_fu_480_s_buf_d1;
    end else begin
        vs_buf_d1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_buf_we1 = grp_load_quantize_vit_v_fu_679_vs_buf_we1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_buf_we1 = grp_quantize_v_group_fu_568_vs_buf_we1;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_buf_we1 = grp_load_llm_v_cache_qs_fu_480_s_buf_we1;
    end else begin
        vs_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        vs_cache_i_stream_TREADY = grp_load_llm_v_cache_qs_fu_480_vs_cache_i_stream_TREADY;
    end else begin
        vs_cache_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
        vs_cache_o_stream_TDATA = grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TDATA;
    end else if (((grp_quantize_v_group_fu_568_vs_cache_o_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        vs_cache_o_stream_TDATA = grp_quantize_v_group_fu_568_vs_cache_o_stream_TDATA;
    end else begin
        vs_cache_o_stream_TDATA = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        vs_cache_o_stream_TVALID = grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        vs_cache_o_stream_TVALID = grp_quantize_v_group_fu_568_vs_cache_o_stream_TVALID;
    end else begin
        vs_cache_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((mode_val_read_read_fu_452_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((1'b1 == ap_CS_fsm_state5) & ((mode_val_read_read_fu_452_p2 == 1'd1) | (icmp_ln410_fu_856_p2 == 1'd1)))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state6 : begin
            if (((grp_load_llm_v_cache_qs_fu_480_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state6))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state7 : begin
            if (((icmp_ln194_fu_868_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state7))) begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state8 : begin
            if (((grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state8))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            if (((grp_quantize_v_group_fu_568_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        ap_ST_fsm_state11 : begin
            if (((icmp_ln414_fu_1140_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state11))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end
        end
        ap_ST_fsm_state12 : begin
            if (((grp_shared_rv_bmm_head_fu_653_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state12))) begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end
        end
        ap_ST_fsm_state13 : begin
            if (((icmp_ln404_fu_1159_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state13))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end
        end
        ap_ST_fsm_state14 : begin
            if (((grp_load_quantize_vit_v_fu_679_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state14))) begin
                ap_NS_fsm = ap_ST_fsm_state15;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end
        end
        ap_ST_fsm_state15 : begin
            ap_NS_fsm = ap_ST_fsm_state16;
        end
        ap_ST_fsm_state16 : begin
            if (((grp_shared_rv_bmm_head_fu_653_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state16))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state16;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln102_fu_808_p2 = (zext_ln86_fu_800_p1 + 32'd7);

assign add_ln86_fu_758_p2 = (trunc_ln86_fu_755_p1 + 31'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state12 = ap_CS_fsm[32'd11];

assign ap_CS_fsm_state13 = ap_CS_fsm[32'd12];

assign ap_CS_fsm_state14 = ap_CS_fsm[32'd13];

assign ap_CS_fsm_state15 = ap_CS_fsm[32'd14];

assign ap_CS_fsm_state16 = ap_CS_fsm[32'd15];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign aq_stream_TDATA = grp_shared_rv_bmm_head_fu_653_aq_stream_TDATA;

assign as_stream_TDATA = grp_shared_rv_bmm_head_fu_653_as_stream_TDATA;

assign decode_pos_chunk_fu_740_p3 = ((tmp_reg_1566[0:0] == 1'b1) ? sub_ln78_1_fu_726_p2 : trunc_ln78_2_fu_731_p4);

assign decoded_chunk_fu_747_p3 = ((mode_val[0:0] == 1'b1) ? 10'd0 : decode_pos_chunk_fu_740_p3);

assign empty_fu_698_p1 = pos_r[12:0];

assign grp_load_llm_v_cache_qs_fu_480_ap_start = grp_load_llm_v_cache_qs_fu_480_ap_start_reg;

assign grp_load_quantize_vit_v_fu_679_ap_start = grp_load_quantize_vit_v_fu_679_ap_start_reg;

assign grp_load_quantize_vit_v_fu_679_vq_cache_o_stream_TREADY = (vq_cache_o_stream_TREADY & ap_CS_fsm_state14);

assign grp_load_quantize_vit_v_fu_679_vs_cache_o_stream_TREADY = (vs_cache_o_stream_TREADY & ap_CS_fsm_state14);

assign grp_quantize_v_group_fu_568_ap_start = grp_quantize_v_group_fu_568_ap_start_reg;

assign grp_quantize_v_group_fu_568_vq_cache_o_stream_TREADY = (vq_cache_o_stream_TREADY & ap_CS_fsm_state10);

assign grp_quantize_v_group_fu_568_vs_cache_o_stream_TREADY = (vs_cache_o_stream_TREADY & ap_CS_fsm_state10);

assign grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start = grp_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2_fu_498_ap_start_reg;

assign grp_shared_rv_bmm_head_fu_653_ap_start = grp_shared_rv_bmm_head_fu_653_ap_start_reg;

assign grp_shared_rv_bmm_head_fu_653_aq_stream_TREADY = ((aq_stream_TREADY & ap_CS_fsm_state16) | (aq_stream_TREADY & ap_CS_fsm_state12));

assign grp_shared_rv_bmm_head_fu_653_as_stream_TREADY = ((as_stream_TREADY & ap_CS_fsm_state16) | (as_stream_TREADY & ap_CS_fsm_state12));

assign h_2_fu_1165_p2 = (h_fu_442 + 4'd1);

assign hct_1_fu_874_p2 = (hct_reg_458 + 4'd1);

assign icmp_ln194_fu_868_p2 = ((hct_reg_458 == 4'd8) ? 1'b1 : 1'b0);

assign icmp_ln404_fu_1159_p2 = ((h_fu_442 == 4'd12) ? 1'b1 : 1'b0);

assign icmp_ln410_fu_856_p2 = ((kv_fu_438 == 3'd5) ? 1'b1 : 1'b0);

assign icmp_ln414_fu_1140_p2 = ((r_reg_469 == 2'd3) ? 1'b1 : 1'b0);

assign icmp_ln90_fu_803_p2 = ((valid_s_reg_1605 > 31'd1024) ? 1'b1 : 1'b0);

assign kv_2_fu_862_p2 = (kv_fu_438 + 3'd1);

assign mode_val_read_read_fu_452_p2 = mode_val;

assign r_1_fu_1146_p2 = (r_reg_469 + 2'd1);

assign select_ln102_fu_837_p3 = ((tmp_1_fu_814_p3[0:0] == 1'b1) ? sub_ln102_1_fu_822_p2 : trunc_ln102_3_fu_827_p4);

assign sub_ln102_1_fu_822_p2 = (8'd0 - trunc_ln102_2_reg_1611);

assign sub_ln102_fu_784_p2 = ($signed(11'd2041) - $signed(trunc_ln86_1_fu_780_p1));

assign sub_ln78_1_fu_726_p2 = (10'd0 - trunc_ln78_1_reg_1572);

assign sub_ln78_fu_710_p2 = (13'd0 - empty_fu_698_p1);

assign tmp_1_fu_814_p3 = add_ln102_fu_808_p2[32'd31];

assign trunc_ln102_3_fu_827_p4 = {{add_ln102_fu_808_p2[10:3]}};

assign trunc_ln194_fu_880_p1 = hct_reg_458[2:0];

assign trunc_ln78_2_fu_731_p4 = {{pos_r[12:3]}};

assign trunc_ln86_1_fu_780_p1 = valid_s_fu_774_p3[10:0];

assign trunc_ln86_fu_755_p1 = pos_r[30:0];

assign valid_s_fu_774_p3 = ((tmp_reg_1566[0:0] == 1'b1) ? 31'd1 : add_ln86_reg_1593);

assign valid_st_fu_845_p3 = ((icmp_ln90_fu_803_p2[0:0] == 1'b1) ? 8'd128 : select_ln102_fu_837_p3);

assign zext_ln86_fu_800_p1 = valid_s_reg_1605;

endmodule //RV_GEMM_rv_gemm_one_layer
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        vq_cache_i_stream_TVALID,
        vs_cache_i_stream_TVALID,
        valid_st,
        zext_ln119,
        s_buf_address0,
        s_buf_ce0,
        s_buf_q0,
        s_buf_address1,
        s_buf_ce1,
        s_buf_we1,
        s_buf_d1,
        vq_cache_i_stream_TDATA,
        vq_cache_i_stream_TREADY,
        vs_cache_i_stream_TDATA,
        vs_cache_i_stream_TREADY,
        lshr_ln,
        q_buf_0_address1,
        q_buf_0_ce1,
        q_buf_0_we1,
        q_buf_0_d1,
        q_buf_1_address1,
        q_buf_1_ce1,
        q_buf_1_we1,
        q_buf_1_d1,
        q_buf_2_address1,
        q_buf_2_ce1,
        q_buf_2_we1,
        q_buf_2_d1,
        q_buf_3_address1,
        q_buf_3_ce1,
        q_buf_3_we1,
        q_buf_3_d1,
        q_buf_4_address1,
        q_buf_4_ce1,
        q_buf_4_we1,
        q_buf_4_d1,
        q_buf_5_address1,
        q_buf_5_ce1,
        q_buf_5_we1,
        q_buf_5_d1,
        q_buf_6_address1,
        q_buf_6_ce1,
        q_buf_6_we1,
        q_buf_6_d1,
        q_buf_7_address1,
        q_buf_7_ce1,
        q_buf_7_we1,
        q_buf_7_d1,
        tmp_105,
        empty,
        tmp_107
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter4_fsm_state5 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   vq_cache_i_stream_TVALID;
input   vs_cache_i_stream_TVALID;
input  [7:0] valid_st;
input  [9:0] zext_ln119;
output  [9:0] s_buf_address0;
output   s_buf_ce0;
input  [31:0] s_buf_q0;
output  [9:0] s_buf_address1;
output   s_buf_ce1;
output   s_buf_we1;
output  [31:0] s_buf_d1;
input  [63:0] vq_cache_i_stream_TDATA;
output   vq_cache_i_stream_TREADY;
input  [7:0] vs_cache_i_stream_TDATA;
output   vs_cache_i_stream_TREADY;
input  [2:0] lshr_ln;
output  [9:0] q_buf_0_address1;
output   q_buf_0_ce1;
output  [7:0] q_buf_0_we1;
output  [63:0] q_buf_0_d1;
output  [9:0] q_buf_1_address1;
output   q_buf_1_ce1;
output  [7:0] q_buf_1_we1;
output  [63:0] q_buf_1_d1;
output  [9:0] q_buf_2_address1;
output   q_buf_2_ce1;
output  [7:0] q_buf_2_we1;
output  [63:0] q_buf_2_d1;
output  [9:0] q_buf_3_address1;
output   q_buf_3_ce1;
output  [7:0] q_buf_3_we1;
output  [63:0] q_buf_3_d1;
output  [9:0] q_buf_4_address1;
output   q_buf_4_ce1;
output  [7:0] q_buf_4_we1;
output  [63:0] q_buf_4_d1;
output  [9:0] q_buf_5_address1;
output   q_buf_5_ce1;
output  [7:0] q_buf_5_we1;
output  [63:0] q_buf_5_d1;
output  [9:0] q_buf_6_address1;
output   q_buf_6_ce1;
output  [7:0] q_buf_6_we1;
output  [63:0] q_buf_6_d1;
output  [9:0] q_buf_7_address1;
output   q_buf_7_ce1;
output  [7:0] q_buf_7_we1;
output  [63:0] q_buf_7_d1;
input  [5:0] tmp_105;
input  [2:0] empty;
input  [4:0] tmp_107;

reg ap_idle;
reg s_buf_ce0;
reg s_buf_ce1;
reg s_buf_we1;
reg vq_cache_i_stream_TREADY;
reg vs_cache_i_stream_TREADY;
reg q_buf_0_ce1;
reg[7:0] q_buf_0_we1;
reg q_buf_1_ce1;
reg[7:0] q_buf_1_we1;
reg q_buf_2_ce1;
reg[7:0] q_buf_2_we1;
reg q_buf_3_ce1;
reg[7:0] q_buf_3_we1;
reg q_buf_4_ce1;
reg[7:0] q_buf_4_we1;
reg q_buf_5_ce1;
reg[7:0] q_buf_5_we1;
reg q_buf_6_ce1;
reg[7:0] q_buf_6_we1;
reg q_buf_7_ce1;
reg[7:0] q_buf_7_we1;

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
wire   [0:0] icmp_ln112_fu_352_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    vq_cache_i_stream_TDATA_blk_n;
reg    vs_cache_i_stream_TDATA_blk_n;
reg   [7:0] st_3_reg_670;
reg   [0:0] icmp_ln112_reg_675;
reg   [0:0] icmp_ln112_reg_675_pp0_iter1_reg;
reg   [0:0] icmp_ln112_reg_675_pp0_iter2_reg;
reg   [0:0] icmp_ln112_reg_675_pp0_iter3_reg;
wire   [7:0] trunc_ln115_fu_364_p1;
reg   [7:0] trunc_ln115_reg_679;
reg   [7:0] trunc_ln115_1_reg_684;
reg   [7:0] trunc_ln115_2_reg_689;
reg   [7:0] trunc_ln115_3_reg_694;
reg   [7:0] trunc_ln115_4_reg_699;
reg   [7:0] trunc_ln115_5_reg_704;
reg   [7:0] trunc_ln115_6_reg_709;
reg   [7:0] trunc_ln115_7_reg_714;
wire   [3:0] trunc_ln116_fu_438_p1;
reg   [3:0] trunc_ln116_reg_719;
reg   [3:0] trunc_ln116_reg_719_pp0_iter1_reg;
reg   [3:0] trunc_ln116_reg_719_pp0_iter2_reg;
reg   [9:0] s_buf_addr_reg_724;
reg   [9:0] s_buf_addr_reg_724_pp0_iter2_reg;
reg   [9:0] s_buf_addr_reg_724_pp0_iter3_reg;
wire   [63:0] shl_ln119_fu_466_p2;
reg   [63:0] shl_ln119_reg_730;
wire   [63:0] shl_ln119_2_fu_475_p2;
reg   [63:0] shl_ln119_2_reg_735;
wire   [63:0] shl_ln119_3_fu_484_p2;
reg   [63:0] shl_ln119_3_reg_740;
wire   [63:0] shl_ln119_4_fu_493_p2;
reg   [63:0] shl_ln119_4_reg_745;
wire   [63:0] shl_ln119_5_fu_502_p2;
reg   [63:0] shl_ln119_5_reg_750;
wire   [63:0] shl_ln119_6_fu_511_p2;
reg   [63:0] shl_ln119_6_reg_755;
wire   [63:0] shl_ln119_7_fu_520_p2;
reg   [63:0] shl_ln119_7_reg_760;
wire   [63:0] shl_ln119_8_fu_529_p2;
reg   [63:0] shl_ln119_8_reg_765;
wire   [31:0] or_ln121_fu_625_p2;
reg   [31:0] or_ln121_reg_770;
wire   [63:0] zext_ln121_1_fu_455_p1;
wire   [63:0] zext_ln119_1_fu_555_p1;
reg   [10:0] s_base_fu_138;
wire   [10:0] s_base_1_fu_584_p2;
wire    ap_loop_init;
reg   [7:0] st_fu_142;
wire   [7:0] add_ln112_fu_358_p2;
reg   [7:0] ap_sig_allocacmp_st_3;
wire   [7:0] shl_ln119_1_fu_570_p2;
wire   [9:0] zext_ln121_fu_447_p1;
wire   [9:0] add_ln121_fu_450_p2;
wire   [63:0] zext_ln119_3_fu_463_p1;
wire   [63:0] zext_ln119_2_fu_460_p1;
wire   [63:0] zext_ln119_5_fu_472_p1;
wire   [63:0] zext_ln119_6_fu_481_p1;
wire   [63:0] zext_ln119_7_fu_490_p1;
wire   [63:0] zext_ln119_8_fu_499_p1;
wire   [63:0] zext_ln119_9_fu_508_p1;
wire   [63:0] zext_ln119_10_fu_517_p1;
wire   [63:0] zext_ln119_11_fu_526_p1;
wire   [6:0] lshr_ln1_fu_538_p4;
wire   [9:0] tmp_s_fu_548_p3;
wire   [7:0] zext_ln119_4_fu_567_p1;
wire   [31:0] zext_ln121_2_fu_595_p1;
wire   [31:0] shl_ln121_fu_598_p2;
wire   [31:0] xor_ln121_fu_604_p2;
wire   [31:0] zext_ln121_3_fu_616_p1;
wire   [31:0] and_ln121_fu_610_p2;
wire   [31:0] shl_ln121_1_fu_619_p2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 ap_CS_iter4_fsm = 2'd1;
//#0 s_base_fu_138 = 11'd0;
//#0 st_fu_142 = 8'd0;
//#0 ap_done_reg = 1'b0;
end

RV_GEMM_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter4_fsm_state5) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter4_fsm_state5) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= 1'b0;
    end else if ((1'b1 == ap_CS_iter3_fsm_state4)) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        s_base_fu_138 <= 11'd0;
    end else if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        s_base_fu_138 <= s_base_1_fu_584_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        if ((icmp_ln112_fu_352_p2 == 1'd0)) begin
            st_fu_142 <= add_ln112_fu_358_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            st_fu_142 <= 8'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln112_reg_675 <= icmp_ln112_fu_352_p2;
        st_3_reg_670 <= ap_sig_allocacmp_st_3;
        trunc_ln115_1_reg_684 <= {{vq_cache_i_stream_TDATA[15:8]}};
        trunc_ln115_2_reg_689 <= {{vq_cache_i_stream_TDATA[23:16]}};
        trunc_ln115_3_reg_694 <= {{vq_cache_i_stream_TDATA[31:24]}};
        trunc_ln115_4_reg_699 <= {{vq_cache_i_stream_TDATA[39:32]}};
        trunc_ln115_5_reg_704 <= {{vq_cache_i_stream_TDATA[47:40]}};
        trunc_ln115_6_reg_709 <= {{vq_cache_i_stream_TDATA[55:48]}};
        trunc_ln115_7_reg_714 <= {{vq_cache_i_stream_TDATA[63:56]}};
        trunc_ln115_reg_679 <= trunc_ln115_fu_364_p1;
        trunc_ln116_reg_719 <= trunc_ln116_fu_438_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln112_reg_675_pp0_iter1_reg <= icmp_ln112_reg_675;
        s_buf_addr_reg_724 <= zext_ln121_1_fu_455_p1;
        shl_ln119_2_reg_735 <= shl_ln119_2_fu_475_p2;
        shl_ln119_3_reg_740 <= shl_ln119_3_fu_484_p2;
        shl_ln119_4_reg_745 <= shl_ln119_4_fu_493_p2;
        shl_ln119_5_reg_750 <= shl_ln119_5_fu_502_p2;
        shl_ln119_6_reg_755 <= shl_ln119_6_fu_511_p2;
        shl_ln119_7_reg_760 <= shl_ln119_7_fu_520_p2;
        shl_ln119_8_reg_765 <= shl_ln119_8_fu_529_p2;
        shl_ln119_reg_730 <= shl_ln119_fu_466_p2;
        trunc_ln116_reg_719_pp0_iter1_reg <= trunc_ln116_reg_719;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln112_reg_675_pp0_iter2_reg <= icmp_ln112_reg_675_pp0_iter1_reg;
        s_buf_addr_reg_724_pp0_iter2_reg <= s_buf_addr_reg_724;
        trunc_ln116_reg_719_pp0_iter2_reg <= trunc_ln116_reg_719_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter3_fsm_state4)) begin
        icmp_ln112_reg_675_pp0_iter3_reg <= icmp_ln112_reg_675_pp0_iter2_reg;
        or_ln121_reg_770 <= or_ln121_fu_625_p2;
        s_buf_addr_reg_724_pp0_iter3_reg <= s_buf_addr_reg_724_pp0_iter2_reg;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state2_blk = 1'b0;

assign ap_ST_iter2_fsm_state3_blk = 1'b0;

assign ap_ST_iter3_fsm_state4_blk = 1'b0;

assign ap_ST_iter4_fsm_state5_blk = 1'b0;

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln112_fu_352_p2 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter4_fsm_state5) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_st_3 = 8'd0;
    end else begin
        ap_sig_allocacmp_st_3 = st_fu_142;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_0_ce1 = 1'b1;
    end else begin
        q_buf_0_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_0_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_0_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_1_ce1 = 1'b1;
    end else begin
        q_buf_1_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_1_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_1_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_2_ce1 = 1'b1;
    end else begin
        q_buf_2_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_2_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_2_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_3_ce1 = 1'b1;
    end else begin
        q_buf_3_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_3_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_3_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_4_ce1 = 1'b1;
    end else begin
        q_buf_4_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_4_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_4_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_5_ce1 = 1'b1;
    end else begin
        q_buf_5_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_5_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_5_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_6_ce1 = 1'b1;
    end else begin
        q_buf_6_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_6_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_6_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        q_buf_7_ce1 = 1'b1;
    end else begin
        q_buf_7_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln112_reg_675_pp0_iter1_reg == 1'd0))) begin
        q_buf_7_we1 = shl_ln119_1_fu_570_p2;
    end else begin
        q_buf_7_we1 = 8'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter3_fsm_state4) | (1'b1 == ap_CS_iter2_fsm_state3) | (1'b1 == ap_CS_iter1_fsm_state2))) begin
        s_buf_ce0 = 1'b1;
    end else begin
        s_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter4_fsm_state5)) begin
        s_buf_ce1 = 1'b1;
    end else begin
        s_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter4_fsm_state5) & (icmp_ln112_reg_675_pp0_iter3_reg == 1'd0))) begin
        s_buf_we1 = 1'b1;
    end else begin
        s_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln112_fu_352_p2 == 1'd0) & (ap_start_int == 1'b1))) begin
        vq_cache_i_stream_TDATA_blk_n = vq_cache_i_stream_TVALID;
    end else begin
        vq_cache_i_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln112_fu_352_p2 == 1'd0))) begin
        vq_cache_i_stream_TREADY = 1'b1;
    end else begin
        vq_cache_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln112_fu_352_p2 == 1'd0) & (ap_start_int == 1'b1))) begin
        vs_cache_i_stream_TDATA_blk_n = vs_cache_i_stream_TVALID;
    end else begin
        vs_cache_i_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln112_fu_352_p2 == 1'd0))) begin
        vs_cache_i_stream_TREADY = 1'b1;
    end else begin
        vs_cache_i_stream_TREADY = 1'b0;
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
            if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
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
            if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
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
            if ((1'b0 == ap_CS_iter3_fsm_state4)) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else if (((1'b1 == ap_CS_iter3_fsm_state4) | ((1'b1 == ap_CS_iter4_fsm_state5) & (icmp_ln112_reg_675_pp0_iter3_reg == 1'd1)))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter3_fsm_state4)) begin
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

assign add_ln112_fu_358_p2 = (ap_sig_allocacmp_st_3 + 8'd1);

assign add_ln121_fu_450_p2 = (zext_ln119 + zext_ln121_fu_447_p1);

assign and_ln121_fu_610_p2 = (xor_ln121_fu_604_p2 & s_buf_q0);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state4 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state5 = ap_CS_iter4_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((vs_cache_i_stream_TVALID == 1'b0) & (icmp_ln112_fu_352_p2 == 1'd0)) | ((icmp_ln112_fu_352_p2 == 1'd0) & (vq_cache_i_stream_TVALID == 1'b0)));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln112_fu_352_p2 = ((ap_sig_allocacmp_st_3 == valid_st) ? 1'b1 : 1'b0);

assign lshr_ln1_fu_538_p4 = {{s_base_fu_138[9:3]}};

assign or_ln121_fu_625_p2 = (shl_ln121_1_fu_619_p2 | and_ln121_fu_610_p2);

assign q_buf_0_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_0_d1 = shl_ln119_reg_730;

assign q_buf_1_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_1_d1 = shl_ln119_2_reg_735;

assign q_buf_2_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_2_d1 = shl_ln119_3_reg_740;

assign q_buf_3_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_3_d1 = shl_ln119_4_reg_745;

assign q_buf_4_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_4_d1 = shl_ln119_5_reg_750;

assign q_buf_5_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_5_d1 = shl_ln119_6_reg_755;

assign q_buf_6_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_6_d1 = shl_ln119_7_reg_760;

assign q_buf_7_address1 = zext_ln119_1_fu_555_p1;

assign q_buf_7_d1 = shl_ln119_8_reg_765;

assign s_base_1_fu_584_p2 = (s_base_fu_138 + 11'd8);

assign s_buf_address0 = zext_ln121_1_fu_455_p1;

assign s_buf_address1 = s_buf_addr_reg_724_pp0_iter3_reg;

assign s_buf_d1 = or_ln121_reg_770;

assign shl_ln119_1_fu_570_p2 = 8'd1 << zext_ln119_4_fu_567_p1;

assign shl_ln119_2_fu_475_p2 = zext_ln119_5_fu_472_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_3_fu_484_p2 = zext_ln119_6_fu_481_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_4_fu_493_p2 = zext_ln119_7_fu_490_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_5_fu_502_p2 = zext_ln119_8_fu_499_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_6_fu_511_p2 = zext_ln119_9_fu_508_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_7_fu_520_p2 = zext_ln119_10_fu_517_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_8_fu_529_p2 = zext_ln119_11_fu_526_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln119_fu_466_p2 = zext_ln119_3_fu_463_p1 << zext_ln119_2_fu_460_p1;

assign shl_ln121_1_fu_619_p2 = zext_ln121_3_fu_616_p1 << zext_ln121_2_fu_595_p1;

assign shl_ln121_fu_598_p2 = 32'd15 << zext_ln121_2_fu_595_p1;

assign tmp_s_fu_548_p3 = {{lshr_ln}, {lshr_ln1_fu_538_p4}};

assign trunc_ln115_fu_364_p1 = vq_cache_i_stream_TDATA[7:0];

assign trunc_ln116_fu_438_p1 = vs_cache_i_stream_TDATA[3:0];

assign xor_ln121_fu_604_p2 = (shl_ln121_fu_598_p2 ^ 32'd4294967295);

assign zext_ln119_10_fu_517_p1 = trunc_ln115_6_reg_709;

assign zext_ln119_11_fu_526_p1 = trunc_ln115_7_reg_714;

assign zext_ln119_1_fu_555_p1 = tmp_s_fu_548_p3;

assign zext_ln119_2_fu_460_p1 = tmp_105;

assign zext_ln119_3_fu_463_p1 = trunc_ln115_reg_679;

assign zext_ln119_4_fu_567_p1 = empty;

assign zext_ln119_5_fu_472_p1 = trunc_ln115_1_reg_684;

assign zext_ln119_6_fu_481_p1 = trunc_ln115_2_reg_689;

assign zext_ln119_7_fu_490_p1 = trunc_ln115_3_reg_694;

assign zext_ln119_8_fu_499_p1 = trunc_ln115_4_reg_699;

assign zext_ln119_9_fu_508_p1 = trunc_ln115_5_reg_704;

assign zext_ln121_1_fu_455_p1 = add_ln121_fu_450_p2;

assign zext_ln121_2_fu_595_p1 = tmp_107;

assign zext_ln121_3_fu_616_p1 = trunc_ln116_reg_719_pp0_iter2_reg;

assign zext_ln121_fu_447_p1 = st_3_reg_670;

endmodule //RV_GEMM_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        v_stream_TVALID,
        v_stream_TDATA,
        v_stream_TREADY,
        p_0_0_7_0_0_0_i138_out,
        p_0_0_7_0_0_0_i138_out_ap_vld,
        p_0_0_6_0_0_0_i136_out,
        p_0_0_6_0_0_0_i136_out_ap_vld,
        p_0_0_5_0_0_0_i134_out,
        p_0_0_5_0_0_0_i134_out_ap_vld,
        p_0_0_4_0_0_0_i132_out,
        p_0_0_4_0_0_0_i132_out_ap_vld,
        p_0_0_3_0_0_0_i130_out,
        p_0_0_3_0_0_0_i130_out_ap_vld,
        p_0_0_2_0_0_0_i128_out,
        p_0_0_2_0_0_0_i128_out_ap_vld,
        p_0_0_1_0_0_0_i126_out,
        p_0_0_1_0_0_0_i126_out_ap_vld,
        p_0_0_0_0_0_0_i124_out,
        p_0_0_0_0_0_0_i124_out_ap_vld,
        p_0_0_7_0_0_0_i122_out,
        p_0_0_7_0_0_0_i122_out_ap_vld,
        p_0_0_6_0_0_0_i120_out,
        p_0_0_6_0_0_0_i120_out_ap_vld,
        p_0_0_5_0_0_0_i118_out,
        p_0_0_5_0_0_0_i118_out_ap_vld,
        p_0_0_4_0_0_0_i116_out,
        p_0_0_4_0_0_0_i116_out_ap_vld,
        p_0_0_3_0_0_0_i114_out,
        p_0_0_3_0_0_0_i114_out_ap_vld,
        p_0_0_2_0_0_0_i112_out,
        p_0_0_2_0_0_0_i112_out_ap_vld,
        p_0_0_1_0_0_0_i110_out,
        p_0_0_1_0_0_0_i110_out_ap_vld,
        p_0_0_0_0_0_0_i108_out,
        p_0_0_0_0_0_0_i108_out_ap_vld,
        p_0_0_7_0_0_0_i106_out,
        p_0_0_7_0_0_0_i106_out_ap_vld,
        p_0_0_6_0_0_0_i104_out,
        p_0_0_6_0_0_0_i104_out_ap_vld,
        p_0_0_5_0_0_0_i102_out,
        p_0_0_5_0_0_0_i102_out_ap_vld,
        p_0_0_4_0_0_0_i100_out,
        p_0_0_4_0_0_0_i100_out_ap_vld,
        p_0_0_3_0_0_0_i98_out,
        p_0_0_3_0_0_0_i98_out_ap_vld,
        p_0_0_2_0_0_0_i96_out,
        p_0_0_2_0_0_0_i96_out_ap_vld,
        p_0_0_1_0_0_0_i94_out,
        p_0_0_1_0_0_0_i94_out_ap_vld,
        p_0_0_0_0_0_0_i92_out,
        p_0_0_0_0_0_0_i92_out_ap_vld,
        p_0_0_7_0_0_0_i90_out,
        p_0_0_7_0_0_0_i90_out_ap_vld,
        p_0_0_6_0_0_0_i88_out,
        p_0_0_6_0_0_0_i88_out_ap_vld,
        p_0_0_5_0_0_0_i86_out,
        p_0_0_5_0_0_0_i86_out_ap_vld,
        p_0_0_4_0_0_0_i84_out,
        p_0_0_4_0_0_0_i84_out_ap_vld,
        p_0_0_3_0_0_0_i82_out,
        p_0_0_3_0_0_0_i82_out_ap_vld,
        p_0_0_2_0_0_0_i80_out,
        p_0_0_2_0_0_0_i80_out_ap_vld,
        p_0_0_1_0_0_0_i78_out,
        p_0_0_1_0_0_0_i78_out_ap_vld,
        p_0_0_0_0_0_0_i76_out,
        p_0_0_0_0_0_0_i76_out_ap_vld,
        p_0_0_7_0_0_0_i74_out,
        p_0_0_7_0_0_0_i74_out_ap_vld,
        p_0_0_6_0_0_0_i72_out,
        p_0_0_6_0_0_0_i72_out_ap_vld,
        p_0_0_5_0_0_0_i70_out,
        p_0_0_5_0_0_0_i70_out_ap_vld,
        p_0_0_4_0_0_0_i68_out,
        p_0_0_4_0_0_0_i68_out_ap_vld,
        p_0_0_3_0_0_0_i66_out,
        p_0_0_3_0_0_0_i66_out_ap_vld,
        p_0_0_2_0_0_0_i64_out,
        p_0_0_2_0_0_0_i64_out_ap_vld,
        p_0_0_1_0_0_0_i62_out,
        p_0_0_1_0_0_0_i62_out_ap_vld,
        p_0_0_0_0_0_0_i60_out,
        p_0_0_0_0_0_0_i60_out_ap_vld,
        p_0_0_7_0_0_0_i58_out,
        p_0_0_7_0_0_0_i58_out_ap_vld,
        p_0_0_6_0_0_0_i56_out,
        p_0_0_6_0_0_0_i56_out_ap_vld,
        p_0_0_5_0_0_0_i54_out,
        p_0_0_5_0_0_0_i54_out_ap_vld,
        p_0_0_4_0_0_0_i52_out,
        p_0_0_4_0_0_0_i52_out_ap_vld,
        p_0_0_3_0_0_0_i50_out,
        p_0_0_3_0_0_0_i50_out_ap_vld,
        p_0_0_2_0_0_0_i48_out,
        p_0_0_2_0_0_0_i48_out_ap_vld,
        p_0_0_1_0_0_0_i46_out,
        p_0_0_1_0_0_0_i46_out_ap_vld,
        p_0_0_0_0_0_0_i44_out,
        p_0_0_0_0_0_0_i44_out_ap_vld,
        p_0_0_7_0_0_0_i42_out,
        p_0_0_7_0_0_0_i42_out_ap_vld,
        p_0_0_6_0_0_0_i40_out,
        p_0_0_6_0_0_0_i40_out_ap_vld,
        p_0_0_5_0_0_0_i38_out,
        p_0_0_5_0_0_0_i38_out_ap_vld,
        p_0_0_4_0_0_0_i36_out,
        p_0_0_4_0_0_0_i36_out_ap_vld,
        p_0_0_3_0_0_0_i34_out,
        p_0_0_3_0_0_0_i34_out_ap_vld,
        p_0_0_2_0_0_0_i32_out,
        p_0_0_2_0_0_0_i32_out_ap_vld,
        p_0_0_1_0_0_0_i30_out,
        p_0_0_1_0_0_0_i30_out_ap_vld,
        p_0_0_0_0_0_0_i28_out,
        p_0_0_0_0_0_0_i28_out_ap_vld,
        p_0_0_7_0_0_0_i26_out,
        p_0_0_7_0_0_0_i26_out_ap_vld,
        p_0_0_6_0_0_0_i24_out,
        p_0_0_6_0_0_0_i24_out_ap_vld,
        p_0_0_5_0_0_0_i22_out,
        p_0_0_5_0_0_0_i22_out_ap_vld,
        p_0_0_4_0_0_0_i20_out,
        p_0_0_4_0_0_0_i20_out_ap_vld,
        p_0_0_3_0_0_0_i18_out,
        p_0_0_3_0_0_0_i18_out_ap_vld,
        p_0_0_2_0_0_0_i16_out,
        p_0_0_2_0_0_0_i16_out_ap_vld,
        p_0_0_1_0_0_0_i14_out,
        p_0_0_1_0_0_0_i14_out_ap_vld,
        p_0_0_0_0_0_0_i12_out,
        p_0_0_0_0_0_0_i12_out_ap_vld
);

parameter    ap_ST_fsm_state1 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   v_stream_TVALID;
input  [183:0] v_stream_TDATA;
output   v_stream_TREADY;
output  [22:0] p_0_0_7_0_0_0_i138_out;
output   p_0_0_7_0_0_0_i138_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i136_out;
output   p_0_0_6_0_0_0_i136_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i134_out;
output   p_0_0_5_0_0_0_i134_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i132_out;
output   p_0_0_4_0_0_0_i132_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i130_out;
output   p_0_0_3_0_0_0_i130_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i128_out;
output   p_0_0_2_0_0_0_i128_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i126_out;
output   p_0_0_1_0_0_0_i126_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i124_out;
output   p_0_0_0_0_0_0_i124_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i122_out;
output   p_0_0_7_0_0_0_i122_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i120_out;
output   p_0_0_6_0_0_0_i120_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i118_out;
output   p_0_0_5_0_0_0_i118_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i116_out;
output   p_0_0_4_0_0_0_i116_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i114_out;
output   p_0_0_3_0_0_0_i114_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i112_out;
output   p_0_0_2_0_0_0_i112_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i110_out;
output   p_0_0_1_0_0_0_i110_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i108_out;
output   p_0_0_0_0_0_0_i108_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i106_out;
output   p_0_0_7_0_0_0_i106_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i104_out;
output   p_0_0_6_0_0_0_i104_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i102_out;
output   p_0_0_5_0_0_0_i102_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i100_out;
output   p_0_0_4_0_0_0_i100_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i98_out;
output   p_0_0_3_0_0_0_i98_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i96_out;
output   p_0_0_2_0_0_0_i96_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i94_out;
output   p_0_0_1_0_0_0_i94_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i92_out;
output   p_0_0_0_0_0_0_i92_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i90_out;
output   p_0_0_7_0_0_0_i90_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i88_out;
output   p_0_0_6_0_0_0_i88_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i86_out;
output   p_0_0_5_0_0_0_i86_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i84_out;
output   p_0_0_4_0_0_0_i84_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i82_out;
output   p_0_0_3_0_0_0_i82_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i80_out;
output   p_0_0_2_0_0_0_i80_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i78_out;
output   p_0_0_1_0_0_0_i78_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i76_out;
output   p_0_0_0_0_0_0_i76_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i74_out;
output   p_0_0_7_0_0_0_i74_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i72_out;
output   p_0_0_6_0_0_0_i72_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i70_out;
output   p_0_0_5_0_0_0_i70_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i68_out;
output   p_0_0_4_0_0_0_i68_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i66_out;
output   p_0_0_3_0_0_0_i66_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i64_out;
output   p_0_0_2_0_0_0_i64_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i62_out;
output   p_0_0_1_0_0_0_i62_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i60_out;
output   p_0_0_0_0_0_0_i60_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i58_out;
output   p_0_0_7_0_0_0_i58_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i56_out;
output   p_0_0_6_0_0_0_i56_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i54_out;
output   p_0_0_5_0_0_0_i54_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i52_out;
output   p_0_0_4_0_0_0_i52_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i50_out;
output   p_0_0_3_0_0_0_i50_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i48_out;
output   p_0_0_2_0_0_0_i48_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i46_out;
output   p_0_0_1_0_0_0_i46_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i44_out;
output   p_0_0_0_0_0_0_i44_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i42_out;
output   p_0_0_7_0_0_0_i42_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i40_out;
output   p_0_0_6_0_0_0_i40_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i38_out;
output   p_0_0_5_0_0_0_i38_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i36_out;
output   p_0_0_4_0_0_0_i36_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i34_out;
output   p_0_0_3_0_0_0_i34_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i32_out;
output   p_0_0_2_0_0_0_i32_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i30_out;
output   p_0_0_1_0_0_0_i30_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i28_out;
output   p_0_0_0_0_0_0_i28_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0_i26_out;
output   p_0_0_7_0_0_0_i26_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0_i24_out;
output   p_0_0_6_0_0_0_i24_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0_i22_out;
output   p_0_0_5_0_0_0_i22_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0_i20_out;
output   p_0_0_4_0_0_0_i20_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0_i18_out;
output   p_0_0_3_0_0_0_i18_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0_i16_out;
output   p_0_0_2_0_0_0_i16_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0_i14_out;
output   p_0_0_1_0_0_0_i14_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0_i12_out;
output   p_0_0_0_0_0_0_i12_out_ap_vld;

reg ap_idle;
reg v_stream_TREADY;
reg p_0_0_7_0_0_0_i138_out_ap_vld;
reg p_0_0_6_0_0_0_i136_out_ap_vld;
reg p_0_0_5_0_0_0_i134_out_ap_vld;
reg p_0_0_4_0_0_0_i132_out_ap_vld;
reg p_0_0_3_0_0_0_i130_out_ap_vld;
reg p_0_0_2_0_0_0_i128_out_ap_vld;
reg p_0_0_1_0_0_0_i126_out_ap_vld;
reg p_0_0_0_0_0_0_i124_out_ap_vld;
reg p_0_0_7_0_0_0_i122_out_ap_vld;
reg p_0_0_6_0_0_0_i120_out_ap_vld;
reg p_0_0_5_0_0_0_i118_out_ap_vld;
reg p_0_0_4_0_0_0_i116_out_ap_vld;
reg p_0_0_3_0_0_0_i114_out_ap_vld;
reg p_0_0_2_0_0_0_i112_out_ap_vld;
reg p_0_0_1_0_0_0_i110_out_ap_vld;
reg p_0_0_0_0_0_0_i108_out_ap_vld;
reg p_0_0_7_0_0_0_i106_out_ap_vld;
reg p_0_0_6_0_0_0_i104_out_ap_vld;
reg p_0_0_5_0_0_0_i102_out_ap_vld;
reg p_0_0_4_0_0_0_i100_out_ap_vld;
reg p_0_0_3_0_0_0_i98_out_ap_vld;
reg p_0_0_2_0_0_0_i96_out_ap_vld;
reg p_0_0_1_0_0_0_i94_out_ap_vld;
reg p_0_0_0_0_0_0_i92_out_ap_vld;
reg p_0_0_7_0_0_0_i90_out_ap_vld;
reg p_0_0_6_0_0_0_i88_out_ap_vld;
reg p_0_0_5_0_0_0_i86_out_ap_vld;
reg p_0_0_4_0_0_0_i84_out_ap_vld;
reg p_0_0_3_0_0_0_i82_out_ap_vld;
reg p_0_0_2_0_0_0_i80_out_ap_vld;
reg p_0_0_1_0_0_0_i78_out_ap_vld;
reg p_0_0_0_0_0_0_i76_out_ap_vld;
reg p_0_0_7_0_0_0_i74_out_ap_vld;
reg p_0_0_6_0_0_0_i72_out_ap_vld;
reg p_0_0_5_0_0_0_i70_out_ap_vld;
reg p_0_0_4_0_0_0_i68_out_ap_vld;
reg p_0_0_3_0_0_0_i66_out_ap_vld;
reg p_0_0_2_0_0_0_i64_out_ap_vld;
reg p_0_0_1_0_0_0_i62_out_ap_vld;
reg p_0_0_0_0_0_0_i60_out_ap_vld;
reg p_0_0_7_0_0_0_i58_out_ap_vld;
reg p_0_0_6_0_0_0_i56_out_ap_vld;
reg p_0_0_5_0_0_0_i54_out_ap_vld;
reg p_0_0_4_0_0_0_i52_out_ap_vld;
reg p_0_0_3_0_0_0_i50_out_ap_vld;
reg p_0_0_2_0_0_0_i48_out_ap_vld;
reg p_0_0_1_0_0_0_i46_out_ap_vld;
reg p_0_0_0_0_0_0_i44_out_ap_vld;
reg p_0_0_7_0_0_0_i42_out_ap_vld;
reg p_0_0_6_0_0_0_i40_out_ap_vld;
reg p_0_0_5_0_0_0_i38_out_ap_vld;
reg p_0_0_4_0_0_0_i36_out_ap_vld;
reg p_0_0_3_0_0_0_i34_out_ap_vld;
reg p_0_0_2_0_0_0_i32_out_ap_vld;
reg p_0_0_1_0_0_0_i30_out_ap_vld;
reg p_0_0_0_0_0_0_i28_out_ap_vld;
reg p_0_0_7_0_0_0_i26_out_ap_vld;
reg p_0_0_6_0_0_0_i24_out_ap_vld;
reg p_0_0_5_0_0_0_i22_out_ap_vld;
reg p_0_0_4_0_0_0_i20_out_ap_vld;
reg p_0_0_3_0_0_0_i18_out_ap_vld;
reg p_0_0_2_0_0_0_i16_out_ap_vld;
reg p_0_0_1_0_0_0_i14_out_ap_vld;
reg p_0_0_0_0_0_0_i12_out_ap_vld;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [0:0] icmp_ln199_fu_930_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    v_stream_TDATA_blk_n;
reg   [3:0] t_fu_208;
wire   [3:0] add_ln199_fu_936_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_t_3;
reg   [22:0] p_0_0_0_0_0_0_i12_fu_212;
wire   [22:0] trunc_ln201_fu_946_p1;
wire   [2:0] trunc_ln199_fu_942_p1;
reg   [22:0] p_0_0_1_0_0_0_i14_fu_216;
reg   [22:0] p_0_0_2_0_0_0_i16_fu_220;
reg   [22:0] p_0_0_3_0_0_0_i18_fu_224;
reg   [22:0] p_0_0_4_0_0_0_i20_fu_228;
reg   [22:0] p_0_0_5_0_0_0_i22_fu_232;
reg   [22:0] p_0_0_6_0_0_0_i24_fu_236;
reg   [22:0] p_0_0_7_0_0_0_i26_fu_240;
reg   [22:0] p_0_0_0_0_0_0_i28_fu_244;
reg   [22:0] p_0_0_1_0_0_0_i30_fu_248;
reg   [22:0] p_0_0_2_0_0_0_i32_fu_252;
reg   [22:0] p_0_0_3_0_0_0_i34_fu_256;
reg   [22:0] p_0_0_4_0_0_0_i36_fu_260;
reg   [22:0] p_0_0_5_0_0_0_i38_fu_264;
reg   [22:0] p_0_0_6_0_0_0_i40_fu_268;
reg   [22:0] p_0_0_7_0_0_0_i42_fu_272;
reg   [22:0] p_0_0_0_0_0_0_i44_fu_276;
reg   [22:0] p_0_0_1_0_0_0_i46_fu_280;
reg   [22:0] p_0_0_2_0_0_0_i48_fu_284;
reg   [22:0] p_0_0_3_0_0_0_i50_fu_288;
reg   [22:0] p_0_0_4_0_0_0_i52_fu_292;
reg   [22:0] p_0_0_5_0_0_0_i54_fu_296;
reg   [22:0] p_0_0_6_0_0_0_i56_fu_300;
reg   [22:0] p_0_0_7_0_0_0_i58_fu_304;
reg   [22:0] p_0_0_0_0_0_0_i60_fu_308;
reg   [22:0] p_0_0_1_0_0_0_i62_fu_312;
reg   [22:0] p_0_0_2_0_0_0_i64_fu_316;
reg   [22:0] p_0_0_3_0_0_0_i66_fu_320;
reg   [22:0] p_0_0_4_0_0_0_i68_fu_324;
reg   [22:0] p_0_0_5_0_0_0_i70_fu_328;
reg   [22:0] p_0_0_6_0_0_0_i72_fu_332;
reg   [22:0] p_0_0_7_0_0_0_i74_fu_336;
reg   [22:0] p_0_0_0_0_0_0_i76_fu_340;
reg   [22:0] p_0_0_1_0_0_0_i78_fu_344;
reg   [22:0] p_0_0_2_0_0_0_i80_fu_348;
reg   [22:0] p_0_0_3_0_0_0_i82_fu_352;
reg   [22:0] p_0_0_4_0_0_0_i84_fu_356;
reg   [22:0] p_0_0_5_0_0_0_i86_fu_360;
reg   [22:0] p_0_0_6_0_0_0_i88_fu_364;
reg   [22:0] p_0_0_7_0_0_0_i90_fu_368;
reg   [22:0] p_0_0_0_0_0_0_i92_fu_372;
reg   [22:0] p_0_0_1_0_0_0_i94_fu_376;
reg   [22:0] p_0_0_2_0_0_0_i96_fu_380;
reg   [22:0] p_0_0_3_0_0_0_i98_fu_384;
reg   [22:0] p_0_0_4_0_0_0_i100_fu_388;
reg   [22:0] p_0_0_5_0_0_0_i102_fu_392;
reg   [22:0] p_0_0_6_0_0_0_i104_fu_396;
reg   [22:0] p_0_0_7_0_0_0_i106_fu_400;
reg   [22:0] p_0_0_0_0_0_0_i108_fu_404;
reg   [22:0] p_0_0_1_0_0_0_i110_fu_408;
reg   [22:0] p_0_0_2_0_0_0_i112_fu_412;
reg   [22:0] p_0_0_3_0_0_0_i114_fu_416;
reg   [22:0] p_0_0_4_0_0_0_i116_fu_420;
reg   [22:0] p_0_0_5_0_0_0_i118_fu_424;
reg   [22:0] p_0_0_6_0_0_0_i120_fu_428;
reg   [22:0] p_0_0_7_0_0_0_i122_fu_432;
reg   [22:0] p_0_0_0_0_0_0_i124_fu_436;
reg   [22:0] p_0_0_1_0_0_0_i126_fu_440;
reg   [22:0] p_0_0_2_0_0_0_i128_fu_444;
reg   [22:0] p_0_0_3_0_0_0_i130_fu_448;
reg   [22:0] p_0_0_4_0_0_0_i132_fu_452;
reg   [22:0] p_0_0_5_0_0_0_i134_fu_456;
reg   [22:0] p_0_0_6_0_0_0_i136_fu_460;
reg   [22:0] p_0_0_7_0_0_0_i138_fu_464;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [0:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 1'd1;
//#0 t_fu_208 = 4'd0;
//#0 p_0_0_0_0_0_0_i12_fu_212 = 23'd0;
//#0 p_0_0_1_0_0_0_i14_fu_216 = 23'd0;
//#0 p_0_0_2_0_0_0_i16_fu_220 = 23'd0;
//#0 p_0_0_3_0_0_0_i18_fu_224 = 23'd0;
//#0 p_0_0_4_0_0_0_i20_fu_228 = 23'd0;
//#0 p_0_0_5_0_0_0_i22_fu_232 = 23'd0;
//#0 p_0_0_6_0_0_0_i24_fu_236 = 23'd0;
//#0 p_0_0_7_0_0_0_i26_fu_240 = 23'd0;
//#0 p_0_0_0_0_0_0_i28_fu_244 = 23'd0;
//#0 p_0_0_1_0_0_0_i30_fu_248 = 23'd0;
//#0 p_0_0_2_0_0_0_i32_fu_252 = 23'd0;
//#0 p_0_0_3_0_0_0_i34_fu_256 = 23'd0;
//#0 p_0_0_4_0_0_0_i36_fu_260 = 23'd0;
//#0 p_0_0_5_0_0_0_i38_fu_264 = 23'd0;
//#0 p_0_0_6_0_0_0_i40_fu_268 = 23'd0;
//#0 p_0_0_7_0_0_0_i42_fu_272 = 23'd0;
//#0 p_0_0_0_0_0_0_i44_fu_276 = 23'd0;
//#0 p_0_0_1_0_0_0_i46_fu_280 = 23'd0;
//#0 p_0_0_2_0_0_0_i48_fu_284 = 23'd0;
//#0 p_0_0_3_0_0_0_i50_fu_288 = 23'd0;
//#0 p_0_0_4_0_0_0_i52_fu_292 = 23'd0;
//#0 p_0_0_5_0_0_0_i54_fu_296 = 23'd0;
//#0 p_0_0_6_0_0_0_i56_fu_300 = 23'd0;
//#0 p_0_0_7_0_0_0_i58_fu_304 = 23'd0;
//#0 p_0_0_0_0_0_0_i60_fu_308 = 23'd0;
//#0 p_0_0_1_0_0_0_i62_fu_312 = 23'd0;
//#0 p_0_0_2_0_0_0_i64_fu_316 = 23'd0;
//#0 p_0_0_3_0_0_0_i66_fu_320 = 23'd0;
//#0 p_0_0_4_0_0_0_i68_fu_324 = 23'd0;
//#0 p_0_0_5_0_0_0_i70_fu_328 = 23'd0;
//#0 p_0_0_6_0_0_0_i72_fu_332 = 23'd0;
//#0 p_0_0_7_0_0_0_i74_fu_336 = 23'd0;
//#0 p_0_0_0_0_0_0_i76_fu_340 = 23'd0;
//#0 p_0_0_1_0_0_0_i78_fu_344 = 23'd0;
//#0 p_0_0_2_0_0_0_i80_fu_348 = 23'd0;
//#0 p_0_0_3_0_0_0_i82_fu_352 = 23'd0;
//#0 p_0_0_4_0_0_0_i84_fu_356 = 23'd0;
//#0 p_0_0_5_0_0_0_i86_fu_360 = 23'd0;
//#0 p_0_0_6_0_0_0_i88_fu_364 = 23'd0;
//#0 p_0_0_7_0_0_0_i90_fu_368 = 23'd0;
//#0 p_0_0_0_0_0_0_i92_fu_372 = 23'd0;
//#0 p_0_0_1_0_0_0_i94_fu_376 = 23'd0;
//#0 p_0_0_2_0_0_0_i96_fu_380 = 23'd0;
//#0 p_0_0_3_0_0_0_i98_fu_384 = 23'd0;
//#0 p_0_0_4_0_0_0_i100_fu_388 = 23'd0;
//#0 p_0_0_5_0_0_0_i102_fu_392 = 23'd0;
//#0 p_0_0_6_0_0_0_i104_fu_396 = 23'd0;
//#0 p_0_0_7_0_0_0_i106_fu_400 = 23'd0;
//#0 p_0_0_0_0_0_0_i108_fu_404 = 23'd0;
//#0 p_0_0_1_0_0_0_i110_fu_408 = 23'd0;
//#0 p_0_0_2_0_0_0_i112_fu_412 = 23'd0;
//#0 p_0_0_3_0_0_0_i114_fu_416 = 23'd0;
//#0 p_0_0_4_0_0_0_i116_fu_420 = 23'd0;
//#0 p_0_0_5_0_0_0_i118_fu_424 = 23'd0;
//#0 p_0_0_6_0_0_0_i120_fu_428 = 23'd0;
//#0 p_0_0_7_0_0_0_i122_fu_432 = 23'd0;
//#0 p_0_0_0_0_0_0_i124_fu_436 = 23'd0;
//#0 p_0_0_1_0_0_0_i126_fu_440 = 23'd0;
//#0 p_0_0_2_0_0_0_i128_fu_444 = 23'd0;
//#0 p_0_0_3_0_0_0_i130_fu_448 = 23'd0;
//#0 p_0_0_4_0_0_0_i132_fu_452 = 23'd0;
//#0 p_0_0_5_0_0_0_i134_fu_456 = 23'd0;
//#0 p_0_0_6_0_0_0_i136_fu_460 = 23'd0;
//#0 p_0_0_7_0_0_0_i138_fu_464 = 23'd0;
//#0 ap_done_reg = 1'b0;
end

RV_GEMM_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
        if ((icmp_ln199_fu_930_p2 == 1'd0)) begin
            t_fu_208 <= add_ln199_fu_936_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            t_fu_208 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd0))) begin
        p_0_0_0_0_0_0_i108_fu_404 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i110_fu_408 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i112_fu_412 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i114_fu_416 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i116_fu_420 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i118_fu_424 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i120_fu_428 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i122_fu_432 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd7))) begin
        p_0_0_0_0_0_0_i124_fu_436 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i126_fu_440 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i128_fu_444 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i130_fu_448 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i132_fu_452 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i134_fu_456 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i136_fu_460 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i138_fu_464 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd6))) begin
        p_0_0_0_0_0_0_i12_fu_212 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i14_fu_216 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i16_fu_220 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i18_fu_224 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i20_fu_228 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i22_fu_232 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i24_fu_236 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i26_fu_240 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd5))) begin
        p_0_0_0_0_0_0_i28_fu_244 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i30_fu_248 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i32_fu_252 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i34_fu_256 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i36_fu_260 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i38_fu_264 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i40_fu_268 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i42_fu_272 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd4))) begin
        p_0_0_0_0_0_0_i44_fu_276 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i46_fu_280 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i48_fu_284 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i50_fu_288 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i52_fu_292 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i54_fu_296 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i56_fu_300 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i58_fu_304 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd3))) begin
        p_0_0_0_0_0_0_i60_fu_308 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i62_fu_312 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i64_fu_316 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i66_fu_320 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i68_fu_324 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i70_fu_328 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i72_fu_332 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i74_fu_336 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd2))) begin
        p_0_0_0_0_0_0_i76_fu_340 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i78_fu_344 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i80_fu_348 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i82_fu_352 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i84_fu_356 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i86_fu_360 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i88_fu_364 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i90_fu_368 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln199_fu_942_p1 == 3'd1))) begin
        p_0_0_0_0_0_0_i92_fu_372 <= trunc_ln201_fu_946_p1;
        p_0_0_1_0_0_0_i94_fu_376 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0_i96_fu_380 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0_i98_fu_384 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0_i100_fu_388 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0_i102_fu_392 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0_i104_fu_396 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0_i106_fu_400 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_t_3 = 4'd0;
    end else begin
        ap_sig_allocacmp_t_3 = t_fu_208;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i108_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i108_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i124_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i124_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i12_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i12_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i28_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i28_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i44_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i44_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i60_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i60_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i76_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i76_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0_i92_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0_i92_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i110_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i110_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i126_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i126_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i14_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i14_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i30_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i30_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i46_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i46_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i62_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i62_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i78_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i78_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0_i94_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0_i94_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i112_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i112_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i128_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i128_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i16_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i16_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i32_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i32_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i48_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i48_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i64_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i64_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i80_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i80_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0_i96_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0_i96_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i114_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i114_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i130_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i130_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i18_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i18_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i34_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i34_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i50_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i50_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i66_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i66_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i82_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i82_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0_i98_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0_i98_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i100_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i100_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i116_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i116_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i132_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i132_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i20_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i20_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i36_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i36_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i52_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i52_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i68_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i68_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0_i84_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0_i84_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i102_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i102_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i118_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i118_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i134_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i134_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i22_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i22_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i38_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i38_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i54_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i54_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i70_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i70_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0_i86_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0_i86_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i104_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i104_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i120_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i120_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i136_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i136_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i24_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i24_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i40_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i40_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i56_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i56_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i72_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i72_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0_i88_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0_i88_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i106_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i106_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i122_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i122_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i138_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i138_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i26_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i26_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i42_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i42_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i58_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i58_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i74_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i74_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0_i90_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0_i90_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1))) begin
        v_stream_TDATA_blk_n = v_stream_TVALID;
    end else begin
        v_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln199_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1))) begin
        v_stream_TREADY = 1'b1;
    end else begin
        v_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln199_fu_936_p2 = (ap_sig_allocacmp_t_3 + 4'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln199_fu_930_p2 == 1'd0) & (v_stream_TVALID == 1'b0)));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln199_fu_930_p2 = ((ap_sig_allocacmp_t_3 == 4'd8) ? 1'b1 : 1'b0);

assign p_0_0_0_0_0_0_i108_out = p_0_0_0_0_0_0_i108_fu_404;

assign p_0_0_0_0_0_0_i124_out = p_0_0_0_0_0_0_i124_fu_436;

assign p_0_0_0_0_0_0_i12_out = p_0_0_0_0_0_0_i12_fu_212;

assign p_0_0_0_0_0_0_i28_out = p_0_0_0_0_0_0_i28_fu_244;

assign p_0_0_0_0_0_0_i44_out = p_0_0_0_0_0_0_i44_fu_276;

assign p_0_0_0_0_0_0_i60_out = p_0_0_0_0_0_0_i60_fu_308;

assign p_0_0_0_0_0_0_i76_out = p_0_0_0_0_0_0_i76_fu_340;

assign p_0_0_0_0_0_0_i92_out = p_0_0_0_0_0_0_i92_fu_372;

assign p_0_0_1_0_0_0_i110_out = p_0_0_1_0_0_0_i110_fu_408;

assign p_0_0_1_0_0_0_i126_out = p_0_0_1_0_0_0_i126_fu_440;

assign p_0_0_1_0_0_0_i14_out = p_0_0_1_0_0_0_i14_fu_216;

assign p_0_0_1_0_0_0_i30_out = p_0_0_1_0_0_0_i30_fu_248;

assign p_0_0_1_0_0_0_i46_out = p_0_0_1_0_0_0_i46_fu_280;

assign p_0_0_1_0_0_0_i62_out = p_0_0_1_0_0_0_i62_fu_312;

assign p_0_0_1_0_0_0_i78_out = p_0_0_1_0_0_0_i78_fu_344;

assign p_0_0_1_0_0_0_i94_out = p_0_0_1_0_0_0_i94_fu_376;

assign p_0_0_2_0_0_0_i112_out = p_0_0_2_0_0_0_i112_fu_412;

assign p_0_0_2_0_0_0_i128_out = p_0_0_2_0_0_0_i128_fu_444;

assign p_0_0_2_0_0_0_i16_out = p_0_0_2_0_0_0_i16_fu_220;

assign p_0_0_2_0_0_0_i32_out = p_0_0_2_0_0_0_i32_fu_252;

assign p_0_0_2_0_0_0_i48_out = p_0_0_2_0_0_0_i48_fu_284;

assign p_0_0_2_0_0_0_i64_out = p_0_0_2_0_0_0_i64_fu_316;

assign p_0_0_2_0_0_0_i80_out = p_0_0_2_0_0_0_i80_fu_348;

assign p_0_0_2_0_0_0_i96_out = p_0_0_2_0_0_0_i96_fu_380;

assign p_0_0_3_0_0_0_i114_out = p_0_0_3_0_0_0_i114_fu_416;

assign p_0_0_3_0_0_0_i130_out = p_0_0_3_0_0_0_i130_fu_448;

assign p_0_0_3_0_0_0_i18_out = p_0_0_3_0_0_0_i18_fu_224;

assign p_0_0_3_0_0_0_i34_out = p_0_0_3_0_0_0_i34_fu_256;

assign p_0_0_3_0_0_0_i50_out = p_0_0_3_0_0_0_i50_fu_288;

assign p_0_0_3_0_0_0_i66_out = p_0_0_3_0_0_0_i66_fu_320;

assign p_0_0_3_0_0_0_i82_out = p_0_0_3_0_0_0_i82_fu_352;

assign p_0_0_3_0_0_0_i98_out = p_0_0_3_0_0_0_i98_fu_384;

assign p_0_0_4_0_0_0_i100_out = p_0_0_4_0_0_0_i100_fu_388;

assign p_0_0_4_0_0_0_i116_out = p_0_0_4_0_0_0_i116_fu_420;

assign p_0_0_4_0_0_0_i132_out = p_0_0_4_0_0_0_i132_fu_452;

assign p_0_0_4_0_0_0_i20_out = p_0_0_4_0_0_0_i20_fu_228;

assign p_0_0_4_0_0_0_i36_out = p_0_0_4_0_0_0_i36_fu_260;

assign p_0_0_4_0_0_0_i52_out = p_0_0_4_0_0_0_i52_fu_292;

assign p_0_0_4_0_0_0_i68_out = p_0_0_4_0_0_0_i68_fu_324;

assign p_0_0_4_0_0_0_i84_out = p_0_0_4_0_0_0_i84_fu_356;

assign p_0_0_5_0_0_0_i102_out = p_0_0_5_0_0_0_i102_fu_392;

assign p_0_0_5_0_0_0_i118_out = p_0_0_5_0_0_0_i118_fu_424;

assign p_0_0_5_0_0_0_i134_out = p_0_0_5_0_0_0_i134_fu_456;

assign p_0_0_5_0_0_0_i22_out = p_0_0_5_0_0_0_i22_fu_232;

assign p_0_0_5_0_0_0_i38_out = p_0_0_5_0_0_0_i38_fu_264;

assign p_0_0_5_0_0_0_i54_out = p_0_0_5_0_0_0_i54_fu_296;

assign p_0_0_5_0_0_0_i70_out = p_0_0_5_0_0_0_i70_fu_328;

assign p_0_0_5_0_0_0_i86_out = p_0_0_5_0_0_0_i86_fu_360;

assign p_0_0_6_0_0_0_i104_out = p_0_0_6_0_0_0_i104_fu_396;

assign p_0_0_6_0_0_0_i120_out = p_0_0_6_0_0_0_i120_fu_428;

assign p_0_0_6_0_0_0_i136_out = p_0_0_6_0_0_0_i136_fu_460;

assign p_0_0_6_0_0_0_i24_out = p_0_0_6_0_0_0_i24_fu_236;

assign p_0_0_6_0_0_0_i40_out = p_0_0_6_0_0_0_i40_fu_268;

assign p_0_0_6_0_0_0_i56_out = p_0_0_6_0_0_0_i56_fu_300;

assign p_0_0_6_0_0_0_i72_out = p_0_0_6_0_0_0_i72_fu_332;

assign p_0_0_6_0_0_0_i88_out = p_0_0_6_0_0_0_i88_fu_364;

assign p_0_0_7_0_0_0_i106_out = p_0_0_7_0_0_0_i106_fu_400;

assign p_0_0_7_0_0_0_i122_out = p_0_0_7_0_0_0_i122_fu_432;

assign p_0_0_7_0_0_0_i138_out = p_0_0_7_0_0_0_i138_fu_464;

assign p_0_0_7_0_0_0_i26_out = p_0_0_7_0_0_0_i26_fu_240;

assign p_0_0_7_0_0_0_i42_out = p_0_0_7_0_0_0_i42_fu_272;

assign p_0_0_7_0_0_0_i58_out = p_0_0_7_0_0_0_i58_fu_304;

assign p_0_0_7_0_0_0_i74_out = p_0_0_7_0_0_0_i74_fu_336;

assign p_0_0_7_0_0_0_i90_out = p_0_0_7_0_0_0_i90_fu_368;

assign trunc_ln199_fu_942_p1 = ap_sig_allocacmp_t_3[2:0];

assign trunc_ln201_fu_946_p1 = v_stream_TDATA[22:0];

endmodule //RV_GEMM_rv_gemm_one_layer_Pipeline_VITIS_LOOP_199_2
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_load_llm_v_cache_qs (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        vq_cache_i_stream_TDATA,
        vq_cache_i_stream_TVALID,
        vq_cache_i_stream_TREADY,
        vs_cache_i_stream_TDATA,
        vs_cache_i_stream_TVALID,
        vs_cache_i_stream_TREADY,
        q_buf_0_address1,
        q_buf_0_ce1,
        q_buf_0_we1,
        q_buf_0_d1,
        q_buf_1_address1,
        q_buf_1_ce1,
        q_buf_1_we1,
        q_buf_1_d1,
        q_buf_2_address1,
        q_buf_2_ce1,
        q_buf_2_we1,
        q_buf_2_d1,
        q_buf_3_address1,
        q_buf_3_ce1,
        q_buf_3_we1,
        q_buf_3_d1,
        q_buf_4_address1,
        q_buf_4_ce1,
        q_buf_4_we1,
        q_buf_4_d1,
        q_buf_5_address1,
        q_buf_5_ce1,
        q_buf_5_we1,
        q_buf_5_d1,
        q_buf_6_address1,
        q_buf_6_ce1,
        q_buf_6_we1,
        q_buf_6_d1,
        q_buf_7_address1,
        q_buf_7_ce1,
        q_buf_7_we1,
        q_buf_7_d1,
        s_buf_address0,
        s_buf_ce0,
        s_buf_q0,
        s_buf_address1,
        s_buf_ce1,
        s_buf_we1,
        s_buf_d1,
        valid_st
);

parameter    ap_ST_fsm_state1 = 4'd1;
parameter    ap_ST_fsm_state2 = 4'd2;
parameter    ap_ST_fsm_state3 = 4'd4;
parameter    ap_ST_fsm_state4 = 4'd8;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [63:0] vq_cache_i_stream_TDATA;
input   vq_cache_i_stream_TVALID;
output   vq_cache_i_stream_TREADY;
input  [7:0] vs_cache_i_stream_TDATA;
input   vs_cache_i_stream_TVALID;
output   vs_cache_i_stream_TREADY;
output  [9:0] q_buf_0_address1;
output   q_buf_0_ce1;
output  [7:0] q_buf_0_we1;
output  [63:0] q_buf_0_d1;
output  [9:0] q_buf_1_address1;
output   q_buf_1_ce1;
output  [7:0] q_buf_1_we1;
output  [63:0] q_buf_1_d1;
output  [9:0] q_buf_2_address1;
output   q_buf_2_ce1;
output  [7:0] q_buf_2_we1;
output  [63:0] q_buf_2_d1;
output  [9:0] q_buf_3_address1;
output   q_buf_3_ce1;
output  [7:0] q_buf_3_we1;
output  [63:0] q_buf_3_d1;
output  [9:0] q_buf_4_address1;
output   q_buf_4_ce1;
output  [7:0] q_buf_4_we1;
output  [63:0] q_buf_4_d1;
output  [9:0] q_buf_5_address1;
output   q_buf_5_ce1;
output  [7:0] q_buf_5_we1;
output  [63:0] q_buf_5_d1;
output  [9:0] q_buf_6_address1;
output   q_buf_6_ce1;
output  [7:0] q_buf_6_we1;
output  [63:0] q_buf_6_d1;
output  [9:0] q_buf_7_address1;
output   q_buf_7_ce1;
output  [7:0] q_buf_7_we1;
output  [63:0] q_buf_7_d1;
output  [9:0] s_buf_address0;
output   s_buf_ce0;
input  [31:0] s_buf_q0;
output  [9:0] s_buf_address1;
output   s_buf_ce1;
output   s_buf_we1;
output  [31:0] s_buf_d1;
input  [7:0] valid_st;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg vq_cache_i_stream_TREADY;
reg vs_cache_i_stream_TREADY;

(* fsm_encoding = "none" *) reg   [3:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [2:0] lshr_ln_reg_206;
wire    ap_CS_fsm_state2;
wire   [2:0] empty_fu_158_p1;
reg   [2:0] empty_reg_212;
wire   [9:0] tmp_fu_167_p3;
reg   [9:0] tmp_reg_219;
wire    ap_CS_fsm_state3;
wire   [5:0] tmp_s_fu_175_p3;
reg   [5:0] tmp_s_reg_224;
wire   [4:0] tmp_56_fu_183_p3;
reg   [4:0] tmp_56_reg_229;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_done;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_idle;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_ready;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_address0;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_ce0;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_ce1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_we1;
wire   [31:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_d1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_vq_cache_i_stream_TREADY;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_vs_cache_i_stream_TREADY;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_d1;
wire   [9:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_address1;
wire    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_ce1;
wire   [7:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_we1;
wire   [63:0] grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_d1;
reg    grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start_reg;
wire    ap_CS_fsm_state4;
reg   [6:0] c_fu_86;
wire   [6:0] add_ln109_fu_142_p2;
wire   [0:0] icmp_ln109_fu_136_p2;
reg   [3:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 4'd1;
//#0 grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start_reg = 1'b0;
//#0 c_fu_86 = 7'd0;
end

RV_GEMM_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2 grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start),
    .ap_done(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_done),
    .ap_idle(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_idle),
    .ap_ready(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_ready),
    .vq_cache_i_stream_TVALID(vq_cache_i_stream_TVALID),
    .vs_cache_i_stream_TVALID(vs_cache_i_stream_TVALID),
    .valid_st(valid_st),
    .zext_ln119(tmp_reg_219),
    .s_buf_address0(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_address0),
    .s_buf_ce0(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_ce0),
    .s_buf_q0(s_buf_q0),
    .s_buf_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_address1),
    .s_buf_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_ce1),
    .s_buf_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_we1),
    .s_buf_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_d1),
    .vq_cache_i_stream_TDATA(vq_cache_i_stream_TDATA),
    .vq_cache_i_stream_TREADY(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_vq_cache_i_stream_TREADY),
    .vs_cache_i_stream_TDATA(vs_cache_i_stream_TDATA),
    .vs_cache_i_stream_TREADY(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_vs_cache_i_stream_TREADY),
    .lshr_ln(lshr_ln_reg_206),
    .q_buf_0_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_address1),
    .q_buf_0_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_ce1),
    .q_buf_0_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_we1),
    .q_buf_0_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_d1),
    .q_buf_1_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_address1),
    .q_buf_1_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_ce1),
    .q_buf_1_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_we1),
    .q_buf_1_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_d1),
    .q_buf_2_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_address1),
    .q_buf_2_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_ce1),
    .q_buf_2_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_we1),
    .q_buf_2_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_d1),
    .q_buf_3_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_address1),
    .q_buf_3_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_ce1),
    .q_buf_3_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_we1),
    .q_buf_3_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_d1),
    .q_buf_4_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_address1),
    .q_buf_4_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_ce1),
    .q_buf_4_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_we1),
    .q_buf_4_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_d1),
    .q_buf_5_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_address1),
    .q_buf_5_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_ce1),
    .q_buf_5_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_we1),
    .q_buf_5_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_d1),
    .q_buf_6_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_address1),
    .q_buf_6_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_ce1),
    .q_buf_6_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_we1),
    .q_buf_6_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_d1),
    .q_buf_7_address1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_address1),
    .q_buf_7_ce1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_ce1),
    .q_buf_7_we1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_we1),
    .q_buf_7_d1(grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_d1),
    .tmp_105(tmp_s_reg_224),
    .empty(empty_reg_212),
    .tmp_107(tmp_56_reg_229)
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
        grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state3)) begin
            grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start_reg <= 1'b1;
        end else if ((grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_ready == 1'b1)) begin
            grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        c_fu_86 <= 7'd0;
    end else if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln109_fu_136_p2 == 1'd0))) begin
        c_fu_86 <= add_ln109_fu_142_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        empty_reg_212 <= empty_fu_158_p1;
        lshr_ln_reg_206 <= {{c_fu_86[5:3]}};
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        tmp_56_reg_229[4 : 2] <= tmp_56_fu_183_p3[4 : 2];
        tmp_reg_219[9 : 7] <= tmp_fu_167_p3[9 : 7];
        tmp_s_reg_224[5 : 3] <= tmp_s_fu_175_p3[5 : 3];
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_done == 1'b0)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)) | ((1'b1 == ap_CS_fsm_state2) & (icmp_ln109_fu_136_p2 == 1'd1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
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
    if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln109_fu_136_p2 == 1'd1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        vq_cache_i_stream_TREADY = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_vq_cache_i_stream_TREADY;
    end else begin
        vq_cache_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        vs_cache_i_stream_TREADY = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_vs_cache_i_stream_TREADY;
    end else begin
        vs_cache_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln109_fu_136_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            if (((1'b1 == ap_CS_fsm_state4) & (grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln109_fu_142_p2 = (c_fu_86 + 7'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign empty_fu_158_p1 = c_fu_86[2:0];

assign grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_ap_start_reg;

assign icmp_ln109_fu_136_p2 = ((c_fu_86 == 7'd64) ? 1'b1 : 1'b0);

assign q_buf_0_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_address1;

assign q_buf_0_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_ce1;

assign q_buf_0_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_d1;

assign q_buf_0_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_0_we1;

assign q_buf_1_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_address1;

assign q_buf_1_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_ce1;

assign q_buf_1_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_d1;

assign q_buf_1_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_1_we1;

assign q_buf_2_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_address1;

assign q_buf_2_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_ce1;

assign q_buf_2_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_d1;

assign q_buf_2_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_2_we1;

assign q_buf_3_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_address1;

assign q_buf_3_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_ce1;

assign q_buf_3_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_d1;

assign q_buf_3_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_3_we1;

assign q_buf_4_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_address1;

assign q_buf_4_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_ce1;

assign q_buf_4_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_d1;

assign q_buf_4_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_4_we1;

assign q_buf_5_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_address1;

assign q_buf_5_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_ce1;

assign q_buf_5_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_d1;

assign q_buf_5_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_5_we1;

assign q_buf_6_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_address1;

assign q_buf_6_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_ce1;

assign q_buf_6_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_d1;

assign q_buf_6_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_6_we1;

assign q_buf_7_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_address1;

assign q_buf_7_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_ce1;

assign q_buf_7_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_d1;

assign q_buf_7_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_q_buf_7_we1;

assign s_buf_address0 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_address0;

assign s_buf_address1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_address1;

assign s_buf_ce0 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_ce0;

assign s_buf_ce1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_ce1;

assign s_buf_d1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_d1;

assign s_buf_we1 = grp_load_llm_v_cache_qs_Pipeline_VITIS_LOOP_112_2_fu_96_s_buf_we1;

assign tmp_56_fu_183_p3 = {{empty_reg_212}, {2'd0}};

assign tmp_fu_167_p3 = {{lshr_ln_reg_206}, {7'd0}};

assign tmp_s_fu_175_p3 = {{empty_reg_212}, {3'd0}};

always @ (posedge ap_clk) begin
    tmp_reg_219[6:0] <= 7'b0000000;
    tmp_s_reg_224[2:0] <= 3'b000;
    tmp_56_reg_229[1:0] <= 2'b00;
end

endmodule //RV_GEMM_load_llm_v_cache_qs
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module RV_GEMM_shared_rv_bmm_head_rs_buf_RAM_2P_LUTRAM_1R1W (
     
    address0, ce0,
    
    q0, 
      
    address1, ce1,
    d1, we1, 
    
     
    reset, clk);

parameter DataWidth = 4;
parameter AddressWidth = 7;
parameter AddressRange = 128;
 
input[AddressWidth-1:0] address0;
input ce0;

output reg[DataWidth-1:0] q0; 
 
input[AddressWidth-1:0] address1;
input ce1;
input[DataWidth-1:0] d1;
input we1; 


input reset;
input clk;

(* ram_style = "distributed"  *)reg [DataWidth-1:0] ram[0:AddressRange-1];


 



always @(posedge clk) 
begin 
    if (ce0) begin
        q0 <= ram[address0];
    end
end 

 
  

always @(posedge clk)  
begin 
    if (ce1) begin
        if (we1) 
            ram[address1] <= d1; 
    end
end 



 
 

endmodule

// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
`timescale 1 ns / 1 ps

(* use_dsp = "yes" *) module RV_GEMM_mac_muladd_8s_8s_16s_17_4_1_DSP48_0(
    input clk,
    input rst,
    input ce,
    input  [8 - 1:0] in0,
    input  [8 - 1:0] in1,
    input  [16 - 1:0] in2,
    output [17 - 1:0]  dout);

wire signed [27 - 1:0]     a;
wire signed [18 - 1:0]     b;
wire signed [48 - 1:0]     c;
wire signed [45 - 1:0]     m;
wire signed [48 - 1:0]     p;
reg  signed [45 - 1:0]     m_reg;
reg  signed [27 - 1:0]     a_reg;
reg  signed [18 - 1:0]     b_reg;
reg  signed [48 - 1:0]     p_reg;

assign a  = $signed(in0);
assign b  = $signed(in1);
assign c  = $signed(in2);

assign m  = a_reg * b_reg;
assign p  = m_reg + c;

always @(posedge clk) begin
    if (ce) begin
        m_reg  <= m;
        a_reg  <= a;
        b_reg  <= b;
        p_reg  <= p;
    end
end

assign dout = p_reg;

endmodule
`timescale 1 ns / 1 ps
module RV_GEMM_mac_muladd_8s_8s_16s_17_4_1(
    clk,
    reset,
    ce,
    din0,
    din1,
    din2,
    dout);

parameter ID = 32'd1;
parameter NUM_STAGE = 32'd1;
parameter din0_WIDTH = 32'd1;
parameter din1_WIDTH = 32'd1;
parameter din2_WIDTH = 32'd1;
parameter dout_WIDTH = 32'd1;
input clk;
input reset;
input ce;
input[din0_WIDTH - 1:0] din0;
input[din1_WIDTH - 1:0] din1;
input[din2_WIDTH - 1:0] din2;
output[dout_WIDTH - 1:0] dout;



RV_GEMM_mac_muladd_8s_8s_16s_17_4_1_DSP48_0 RV_GEMM_mac_muladd_8s_8s_16s_17_4_1_DSP48_0_U(
    .clk( clk ),
    .rst( reset ),
    .ce( ce ),
    .in0( din0 ),
    .in1( din1 ),
    .in2( din2 ),
    .dout( dout ));

endmodule

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_load_quantize_vit_v (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        v_stream_TDATA,
        v_stream_TVALID,
        v_stream_TREADY,
        vq_buf_0_address1,
        vq_buf_0_ce1,
        vq_buf_0_we1,
        vq_buf_0_d1,
        vq_buf_1_address1,
        vq_buf_1_ce1,
        vq_buf_1_we1,
        vq_buf_1_d1,
        vq_buf_2_address1,
        vq_buf_2_ce1,
        vq_buf_2_we1,
        vq_buf_2_d1,
        vq_buf_3_address1,
        vq_buf_3_ce1,
        vq_buf_3_we1,
        vq_buf_3_d1,
        vq_buf_4_address1,
        vq_buf_4_ce1,
        vq_buf_4_we1,
        vq_buf_4_d1,
        vq_buf_5_address1,
        vq_buf_5_ce1,
        vq_buf_5_we1,
        vq_buf_5_d1,
        vq_buf_6_address1,
        vq_buf_6_ce1,
        vq_buf_6_we1,
        vq_buf_6_d1,
        vq_buf_7_address1,
        vq_buf_7_ce1,
        vq_buf_7_we1,
        vq_buf_7_d1,
        vs_buf_address0,
        vs_buf_ce0,
        vs_buf_q0,
        vs_buf_address1,
        vs_buf_ce1,
        vs_buf_we1,
        vs_buf_d1,
        vq_cache_o_stream_TDATA,
        vq_cache_o_stream_TVALID,
        vq_cache_o_stream_TREADY,
        vs_cache_o_stream_TDATA,
        vs_cache_o_stream_TVALID,
        vs_cache_o_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 5'd1;
parameter    ap_ST_fsm_state2 = 5'd2;
parameter    ap_ST_fsm_state3 = 5'd4;
parameter    ap_ST_fsm_state4 = 5'd8;
parameter    ap_ST_fsm_state5 = 5'd16;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [183:0] v_stream_TDATA;
input   v_stream_TVALID;
output   v_stream_TREADY;
output  [9:0] vq_buf_0_address1;
output   vq_buf_0_ce1;
output  [7:0] vq_buf_0_we1;
output  [63:0] vq_buf_0_d1;
output  [9:0] vq_buf_1_address1;
output   vq_buf_1_ce1;
output  [7:0] vq_buf_1_we1;
output  [63:0] vq_buf_1_d1;
output  [9:0] vq_buf_2_address1;
output   vq_buf_2_ce1;
output  [7:0] vq_buf_2_we1;
output  [63:0] vq_buf_2_d1;
output  [9:0] vq_buf_3_address1;
output   vq_buf_3_ce1;
output  [7:0] vq_buf_3_we1;
output  [63:0] vq_buf_3_d1;
output  [9:0] vq_buf_4_address1;
output   vq_buf_4_ce1;
output  [7:0] vq_buf_4_we1;
output  [63:0] vq_buf_4_d1;
output  [9:0] vq_buf_5_address1;
output   vq_buf_5_ce1;
output  [7:0] vq_buf_5_we1;
output  [63:0] vq_buf_5_d1;
output  [9:0] vq_buf_6_address1;
output   vq_buf_6_ce1;
output  [7:0] vq_buf_6_we1;
output  [63:0] vq_buf_6_d1;
output  [9:0] vq_buf_7_address1;
output   vq_buf_7_ce1;
output  [7:0] vq_buf_7_we1;
output  [63:0] vq_buf_7_d1;
output  [9:0] vs_buf_address0;
output   vs_buf_ce0;
input  [31:0] vs_buf_q0;
output  [9:0] vs_buf_address1;
output   vs_buf_ce1;
output   vs_buf_we1;
output  [31:0] vs_buf_d1;
output  [63:0] vq_cache_o_stream_TDATA;
output   vq_cache_o_stream_TVALID;
input   vq_cache_o_stream_TREADY;
output  [7:0] vs_cache_o_stream_TDATA;
output   vs_cache_o_stream_TVALID;
input   vs_cache_o_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg v_stream_TREADY;

(* fsm_encoding = "none" *) reg   [4:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [3:0] select_ln220_fu_569_p3;
reg   [3:0] select_ln220_reg_1272;
wire    ap_CS_fsm_state3;
wire   [7:0] select_ln220_1_fu_577_p3;
reg   [7:0] select_ln220_1_reg_1277;
wire   [2:0] trunc_ln221_fu_585_p1;
reg   [2:0] trunc_ln221_reg_1282;
wire   [9:0] zext_ln220_fu_594_p1;
reg   [9:0] zext_ln220_reg_1287;
wire    ap_CS_fsm_state4;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_done;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_idle;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_ready;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_v_stream_TREADY;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0127_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0127_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0125_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0125_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0123_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0123_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0121_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0121_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0119_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0119_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0117_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0117_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_0115_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_0115_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_0113_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_0113_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0111_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0111_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0109_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0109_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0107_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0107_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0105_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0105_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0103_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0103_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0101_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0101_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_099_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_099_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_097_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_097_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_095_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_095_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_093_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_093_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_091_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_091_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_089_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_089_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_087_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_087_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_085_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_085_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_083_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_083_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_081_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_081_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_079_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_079_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_077_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_077_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_075_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_075_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_073_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_073_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_071_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_071_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_069_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_069_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_067_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_067_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_065_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_065_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_063_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_063_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_061_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_061_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_059_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_059_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_057_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_057_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_055_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_055_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_053_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_053_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_051_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_051_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_049_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_049_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_047_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_047_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_045_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_045_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_043_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_043_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_041_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_041_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_039_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_039_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_037_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_037_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_035_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_035_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_033_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_033_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_031_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_031_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_029_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_029_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_027_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_027_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_025_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_025_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_023_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_023_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_021_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_021_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_019_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_019_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_017_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_017_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_015_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_015_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_013_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_013_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_011_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_011_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_09_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_09_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_07_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_07_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_05_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_05_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_03_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_03_out_ap_vld;
wire   [22:0] grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_01_out;
wire    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_01_out_ap_vld;
wire    grp_quantize_v_group_fu_422_ap_start;
wire    grp_quantize_v_group_fu_422_ap_done;
wire    grp_quantize_v_group_fu_422_ap_idle;
wire    grp_quantize_v_group_fu_422_ap_ready;
wire    grp_quantize_v_group_fu_422_vq_cache_o_stream_TREADY;
wire    grp_quantize_v_group_fu_422_vs_cache_o_stream_TREADY;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_0_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_0_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_0_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_0_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_1_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_1_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_1_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_1_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_2_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_2_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_2_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_2_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_3_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_3_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_3_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_3_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_4_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_4_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_4_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_4_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_5_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_5_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_5_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_5_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_6_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_6_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_6_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_6_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vq_buf_7_address1;
wire    grp_quantize_v_group_fu_422_vq_buf_7_ce1;
wire   [7:0] grp_quantize_v_group_fu_422_vq_buf_7_we1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_buf_7_d1;
wire   [9:0] grp_quantize_v_group_fu_422_vs_buf_address0;
wire    grp_quantize_v_group_fu_422_vs_buf_ce0;
wire   [9:0] grp_quantize_v_group_fu_422_vs_buf_address1;
wire    grp_quantize_v_group_fu_422_vs_buf_ce1;
wire    grp_quantize_v_group_fu_422_vs_buf_we1;
wire   [31:0] grp_quantize_v_group_fu_422_vs_buf_d1;
wire   [63:0] grp_quantize_v_group_fu_422_vq_cache_o_stream_TDATA;
wire    grp_quantize_v_group_fu_422_vq_cache_o_stream_TVALID;
wire   [7:0] grp_quantize_v_group_fu_422_vs_cache_o_stream_TDATA;
wire    grp_quantize_v_group_fu_422_vs_cache_o_stream_TVALID;
reg    grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start_reg;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln220_fu_534_p2;
reg    grp_quantize_v_group_fu_422_ap_start_reg;
wire    ap_CS_fsm_state5;
reg   [3:0] hct_fu_84;
wire   [3:0] hct_2_fu_854_p2;
reg   [7:0] tt_fu_88;
reg   [10:0] indvar_flatten_fu_92;
wire   [10:0] add_ln220_fu_540_p2;
wire   [0:0] icmp_ln221_fu_563_p2;
wire   [7:0] tt_2_fu_557_p2;
reg   [4:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 5'd1;
//#0 grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start_reg = 1'b0;
//#0 grp_quantize_v_group_fu_422_ap_start_reg = 1'b0;
//#0 hct_fu_84 = 4'd0;
//#0 tt_fu_88 = 8'd0;
//#0 indvar_flatten_fu_92 = 11'd0;
end

RV_GEMM_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3 grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start),
    .ap_done(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_done),
    .ap_idle(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_idle),
    .ap_ready(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_ready),
    .v_stream_TVALID(v_stream_TVALID),
    .v_stream_TDATA(v_stream_TDATA),
    .v_stream_TREADY(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_v_stream_TREADY),
    .p_0_0_7_0_0_0127_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0127_out),
    .p_0_0_7_0_0_0127_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0127_out_ap_vld),
    .p_0_0_6_0_0_0125_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0125_out),
    .p_0_0_6_0_0_0125_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0125_out_ap_vld),
    .p_0_0_5_0_0_0123_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0123_out),
    .p_0_0_5_0_0_0123_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0123_out_ap_vld),
    .p_0_0_4_0_0_0121_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0121_out),
    .p_0_0_4_0_0_0121_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0121_out_ap_vld),
    .p_0_0_3_0_0_0119_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0119_out),
    .p_0_0_3_0_0_0119_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0119_out_ap_vld),
    .p_0_0_2_0_0_0117_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0117_out),
    .p_0_0_2_0_0_0117_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0117_out_ap_vld),
    .p_0_0_1_0_0_0115_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_0115_out),
    .p_0_0_1_0_0_0115_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_0115_out_ap_vld),
    .p_0_0_0_0_0_0113_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_0113_out),
    .p_0_0_0_0_0_0113_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_0113_out_ap_vld),
    .p_0_0_7_0_0_0111_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0111_out),
    .p_0_0_7_0_0_0111_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0111_out_ap_vld),
    .p_0_0_6_0_0_0109_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0109_out),
    .p_0_0_6_0_0_0109_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0109_out_ap_vld),
    .p_0_0_5_0_0_0107_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0107_out),
    .p_0_0_5_0_0_0107_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0107_out_ap_vld),
    .p_0_0_4_0_0_0105_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0105_out),
    .p_0_0_4_0_0_0105_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0105_out_ap_vld),
    .p_0_0_3_0_0_0103_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0103_out),
    .p_0_0_3_0_0_0103_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0103_out_ap_vld),
    .p_0_0_2_0_0_0101_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0101_out),
    .p_0_0_2_0_0_0101_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0101_out_ap_vld),
    .p_0_0_1_0_0_099_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_099_out),
    .p_0_0_1_0_0_099_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_099_out_ap_vld),
    .p_0_0_0_0_0_097_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_097_out),
    .p_0_0_0_0_0_097_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_097_out_ap_vld),
    .p_0_0_7_0_0_095_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_095_out),
    .p_0_0_7_0_0_095_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_095_out_ap_vld),
    .p_0_0_6_0_0_093_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_093_out),
    .p_0_0_6_0_0_093_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_093_out_ap_vld),
    .p_0_0_5_0_0_091_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_091_out),
    .p_0_0_5_0_0_091_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_091_out_ap_vld),
    .p_0_0_4_0_0_089_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_089_out),
    .p_0_0_4_0_0_089_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_089_out_ap_vld),
    .p_0_0_3_0_0_087_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_087_out),
    .p_0_0_3_0_0_087_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_087_out_ap_vld),
    .p_0_0_2_0_0_085_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_085_out),
    .p_0_0_2_0_0_085_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_085_out_ap_vld),
    .p_0_0_1_0_0_083_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_083_out),
    .p_0_0_1_0_0_083_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_083_out_ap_vld),
    .p_0_0_0_0_0_081_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_081_out),
    .p_0_0_0_0_0_081_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_081_out_ap_vld),
    .p_0_0_7_0_0_079_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_079_out),
    .p_0_0_7_0_0_079_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_079_out_ap_vld),
    .p_0_0_6_0_0_077_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_077_out),
    .p_0_0_6_0_0_077_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_077_out_ap_vld),
    .p_0_0_5_0_0_075_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_075_out),
    .p_0_0_5_0_0_075_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_075_out_ap_vld),
    .p_0_0_4_0_0_073_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_073_out),
    .p_0_0_4_0_0_073_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_073_out_ap_vld),
    .p_0_0_3_0_0_071_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_071_out),
    .p_0_0_3_0_0_071_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_071_out_ap_vld),
    .p_0_0_2_0_0_069_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_069_out),
    .p_0_0_2_0_0_069_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_069_out_ap_vld),
    .p_0_0_1_0_0_067_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_067_out),
    .p_0_0_1_0_0_067_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_067_out_ap_vld),
    .p_0_0_0_0_0_065_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_065_out),
    .p_0_0_0_0_0_065_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_065_out_ap_vld),
    .p_0_0_7_0_0_063_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_063_out),
    .p_0_0_7_0_0_063_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_063_out_ap_vld),
    .p_0_0_6_0_0_061_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_061_out),
    .p_0_0_6_0_0_061_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_061_out_ap_vld),
    .p_0_0_5_0_0_059_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_059_out),
    .p_0_0_5_0_0_059_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_059_out_ap_vld),
    .p_0_0_4_0_0_057_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_057_out),
    .p_0_0_4_0_0_057_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_057_out_ap_vld),
    .p_0_0_3_0_0_055_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_055_out),
    .p_0_0_3_0_0_055_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_055_out_ap_vld),
    .p_0_0_2_0_0_053_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_053_out),
    .p_0_0_2_0_0_053_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_053_out_ap_vld),
    .p_0_0_1_0_0_051_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_051_out),
    .p_0_0_1_0_0_051_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_051_out_ap_vld),
    .p_0_0_0_0_0_049_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_049_out),
    .p_0_0_0_0_0_049_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_049_out_ap_vld),
    .p_0_0_7_0_0_047_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_047_out),
    .p_0_0_7_0_0_047_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_047_out_ap_vld),
    .p_0_0_6_0_0_045_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_045_out),
    .p_0_0_6_0_0_045_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_045_out_ap_vld),
    .p_0_0_5_0_0_043_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_043_out),
    .p_0_0_5_0_0_043_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_043_out_ap_vld),
    .p_0_0_4_0_0_041_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_041_out),
    .p_0_0_4_0_0_041_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_041_out_ap_vld),
    .p_0_0_3_0_0_039_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_039_out),
    .p_0_0_3_0_0_039_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_039_out_ap_vld),
    .p_0_0_2_0_0_037_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_037_out),
    .p_0_0_2_0_0_037_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_037_out_ap_vld),
    .p_0_0_1_0_0_035_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_035_out),
    .p_0_0_1_0_0_035_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_035_out_ap_vld),
    .p_0_0_0_0_0_033_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_033_out),
    .p_0_0_0_0_0_033_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_033_out_ap_vld),
    .p_0_0_7_0_0_031_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_031_out),
    .p_0_0_7_0_0_031_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_031_out_ap_vld),
    .p_0_0_6_0_0_029_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_029_out),
    .p_0_0_6_0_0_029_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_029_out_ap_vld),
    .p_0_0_5_0_0_027_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_027_out),
    .p_0_0_5_0_0_027_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_027_out_ap_vld),
    .p_0_0_4_0_0_025_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_025_out),
    .p_0_0_4_0_0_025_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_025_out_ap_vld),
    .p_0_0_3_0_0_023_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_023_out),
    .p_0_0_3_0_0_023_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_023_out_ap_vld),
    .p_0_0_2_0_0_021_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_021_out),
    .p_0_0_2_0_0_021_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_021_out_ap_vld),
    .p_0_0_1_0_0_019_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_019_out),
    .p_0_0_1_0_0_019_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_019_out_ap_vld),
    .p_0_0_0_0_0_017_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_017_out),
    .p_0_0_0_0_0_017_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_017_out_ap_vld),
    .p_0_0_7_0_0_015_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_015_out),
    .p_0_0_7_0_0_015_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_015_out_ap_vld),
    .p_0_0_6_0_0_013_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_013_out),
    .p_0_0_6_0_0_013_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_013_out_ap_vld),
    .p_0_0_5_0_0_011_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_011_out),
    .p_0_0_5_0_0_011_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_011_out_ap_vld),
    .p_0_0_4_0_0_09_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_09_out),
    .p_0_0_4_0_0_09_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_09_out_ap_vld),
    .p_0_0_3_0_0_07_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_07_out),
    .p_0_0_3_0_0_07_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_07_out_ap_vld),
    .p_0_0_2_0_0_05_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_05_out),
    .p_0_0_2_0_0_05_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_05_out_ap_vld),
    .p_0_0_1_0_0_03_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_03_out),
    .p_0_0_1_0_0_03_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_03_out_ap_vld),
    .p_0_0_0_0_0_01_out(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_01_out),
    .p_0_0_0_0_0_01_out_ap_vld(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_01_out_ap_vld)
);

RV_GEMM_quantize_v_group grp_quantize_v_group_fu_422(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_quantize_v_group_fu_422_ap_start),
    .ap_done(grp_quantize_v_group_fu_422_ap_done),
    .ap_idle(grp_quantize_v_group_fu_422_ap_idle),
    .ap_ready(grp_quantize_v_group_fu_422_ap_ready),
    .vq_cache_o_stream_TREADY(grp_quantize_v_group_fu_422_vq_cache_o_stream_TREADY),
    .vs_cache_o_stream_TREADY(grp_quantize_v_group_fu_422_vs_cache_o_stream_TREADY),
    .hct(trunc_ln221_reg_1282),
    .st(zext_ln220_reg_1287),
    .raw_tile_0_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_097_out),
    .raw_tile_0_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_081_out),
    .raw_tile_0_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_065_out),
    .raw_tile_0_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_049_out),
    .raw_tile_0_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_033_out),
    .raw_tile_0_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_017_out),
    .raw_tile_0_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_01_out),
    .raw_tile_0_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_0_0_0_0113_out),
    .raw_tile_1_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_099_out),
    .raw_tile_1_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_083_out),
    .raw_tile_1_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_067_out),
    .raw_tile_1_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_051_out),
    .raw_tile_1_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_035_out),
    .raw_tile_1_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_019_out),
    .raw_tile_1_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_03_out),
    .raw_tile_1_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_1_0_0_0115_out),
    .raw_tile_2_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0101_out),
    .raw_tile_2_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_085_out),
    .raw_tile_2_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_069_out),
    .raw_tile_2_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_053_out),
    .raw_tile_2_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_037_out),
    .raw_tile_2_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_021_out),
    .raw_tile_2_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_05_out),
    .raw_tile_2_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_2_0_0_0117_out),
    .raw_tile_3_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0103_out),
    .raw_tile_3_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_087_out),
    .raw_tile_3_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_071_out),
    .raw_tile_3_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_055_out),
    .raw_tile_3_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_039_out),
    .raw_tile_3_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_023_out),
    .raw_tile_3_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_07_out),
    .raw_tile_3_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_3_0_0_0119_out),
    .raw_tile_4_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0105_out),
    .raw_tile_4_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_089_out),
    .raw_tile_4_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_073_out),
    .raw_tile_4_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_057_out),
    .raw_tile_4_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_041_out),
    .raw_tile_4_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_025_out),
    .raw_tile_4_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_09_out),
    .raw_tile_4_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_4_0_0_0121_out),
    .raw_tile_5_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0107_out),
    .raw_tile_5_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_091_out),
    .raw_tile_5_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_075_out),
    .raw_tile_5_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_059_out),
    .raw_tile_5_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_043_out),
    .raw_tile_5_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_027_out),
    .raw_tile_5_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_011_out),
    .raw_tile_5_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_5_0_0_0123_out),
    .raw_tile_6_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0109_out),
    .raw_tile_6_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_093_out),
    .raw_tile_6_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_077_out),
    .raw_tile_6_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_061_out),
    .raw_tile_6_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_045_out),
    .raw_tile_6_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_029_out),
    .raw_tile_6_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_013_out),
    .raw_tile_6_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_6_0_0_0125_out),
    .raw_tile_7_0_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0111_out),
    .raw_tile_7_1_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_095_out),
    .raw_tile_7_2_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_079_out),
    .raw_tile_7_3_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_063_out),
    .raw_tile_7_4_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_047_out),
    .raw_tile_7_5_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_031_out),
    .raw_tile_7_6_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_015_out),
    .raw_tile_7_7_val(grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_p_0_0_7_0_0_0127_out),
    .write_cache_stream(1'd0),
    .vq_buf_0_address1(grp_quantize_v_group_fu_422_vq_buf_0_address1),
    .vq_buf_0_ce1(grp_quantize_v_group_fu_422_vq_buf_0_ce1),
    .vq_buf_0_we1(grp_quantize_v_group_fu_422_vq_buf_0_we1),
    .vq_buf_0_d1(grp_quantize_v_group_fu_422_vq_buf_0_d1),
    .vq_buf_1_address1(grp_quantize_v_group_fu_422_vq_buf_1_address1),
    .vq_buf_1_ce1(grp_quantize_v_group_fu_422_vq_buf_1_ce1),
    .vq_buf_1_we1(grp_quantize_v_group_fu_422_vq_buf_1_we1),
    .vq_buf_1_d1(grp_quantize_v_group_fu_422_vq_buf_1_d1),
    .vq_buf_2_address1(grp_quantize_v_group_fu_422_vq_buf_2_address1),
    .vq_buf_2_ce1(grp_quantize_v_group_fu_422_vq_buf_2_ce1),
    .vq_buf_2_we1(grp_quantize_v_group_fu_422_vq_buf_2_we1),
    .vq_buf_2_d1(grp_quantize_v_group_fu_422_vq_buf_2_d1),
    .vq_buf_3_address1(grp_quantize_v_group_fu_422_vq_buf_3_address1),
    .vq_buf_3_ce1(grp_quantize_v_group_fu_422_vq_buf_3_ce1),
    .vq_buf_3_we1(grp_quantize_v_group_fu_422_vq_buf_3_we1),
    .vq_buf_3_d1(grp_quantize_v_group_fu_422_vq_buf_3_d1),
    .vq_buf_4_address1(grp_quantize_v_group_fu_422_vq_buf_4_address1),
    .vq_buf_4_ce1(grp_quantize_v_group_fu_422_vq_buf_4_ce1),
    .vq_buf_4_we1(grp_quantize_v_group_fu_422_vq_buf_4_we1),
    .vq_buf_4_d1(grp_quantize_v_group_fu_422_vq_buf_4_d1),
    .vq_buf_5_address1(grp_quantize_v_group_fu_422_vq_buf_5_address1),
    .vq_buf_5_ce1(grp_quantize_v_group_fu_422_vq_buf_5_ce1),
    .vq_buf_5_we1(grp_quantize_v_group_fu_422_vq_buf_5_we1),
    .vq_buf_5_d1(grp_quantize_v_group_fu_422_vq_buf_5_d1),
    .vq_buf_6_address1(grp_quantize_v_group_fu_422_vq_buf_6_address1),
    .vq_buf_6_ce1(grp_quantize_v_group_fu_422_vq_buf_6_ce1),
    .vq_buf_6_we1(grp_quantize_v_group_fu_422_vq_buf_6_we1),
    .vq_buf_6_d1(grp_quantize_v_group_fu_422_vq_buf_6_d1),
    .vq_buf_7_address1(grp_quantize_v_group_fu_422_vq_buf_7_address1),
    .vq_buf_7_ce1(grp_quantize_v_group_fu_422_vq_buf_7_ce1),
    .vq_buf_7_we1(grp_quantize_v_group_fu_422_vq_buf_7_we1),
    .vq_buf_7_d1(grp_quantize_v_group_fu_422_vq_buf_7_d1),
    .vs_buf_address0(grp_quantize_v_group_fu_422_vs_buf_address0),
    .vs_buf_ce0(grp_quantize_v_group_fu_422_vs_buf_ce0),
    .vs_buf_q0(vs_buf_q0),
    .vs_buf_address1(grp_quantize_v_group_fu_422_vs_buf_address1),
    .vs_buf_ce1(grp_quantize_v_group_fu_422_vs_buf_ce1),
    .vs_buf_we1(grp_quantize_v_group_fu_422_vs_buf_we1),
    .vs_buf_d1(grp_quantize_v_group_fu_422_vs_buf_d1),
    .vq_cache_o_stream_TDATA(grp_quantize_v_group_fu_422_vq_cache_o_stream_TDATA),
    .vq_cache_o_stream_TVALID(grp_quantize_v_group_fu_422_vq_cache_o_stream_TVALID),
    .vs_cache_o_stream_TDATA(grp_quantize_v_group_fu_422_vs_cache_o_stream_TDATA),
    .vs_cache_o_stream_TVALID(grp_quantize_v_group_fu_422_vs_cache_o_stream_TVALID)
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
        grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start_reg <= 1'b0;
    end else begin
        if (((icmp_ln220_fu_534_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
            grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start_reg <= 1'b1;
        end else if ((grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_ready == 1'b1)) begin
            grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_quantize_v_group_fu_422_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state4)) begin
            grp_quantize_v_group_fu_422_ap_start_reg <= 1'b1;
        end else if ((grp_quantize_v_group_fu_422_ap_ready == 1'b1)) begin
            grp_quantize_v_group_fu_422_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        hct_fu_84 <= 4'd0;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        hct_fu_84 <= hct_2_fu_854_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        indvar_flatten_fu_92 <= 11'd0;
    end else if (((icmp_ln220_fu_534_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        indvar_flatten_fu_92 <= add_ln220_fu_540_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        tt_fu_88 <= 8'd0;
    end else if (((1'b1 == ap_CS_fsm_state3) & (grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_done == 1'b1))) begin
        tt_fu_88 <= select_ln220_1_fu_577_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        select_ln220_1_reg_1277 <= select_ln220_1_fu_577_p3;
        select_ln220_reg_1272 <= select_ln220_fu_569_p3;
        trunc_ln221_reg_1282 <= trunc_ln221_fu_585_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        zext_ln220_reg_1287[7 : 0] <= zext_ln220_fu_594_p1[7 : 0];
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((grp_quantize_v_group_fu_422_ap_done == 1'b0)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((icmp_ln220_fu_534_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2)) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
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
    if (((icmp_ln220_fu_534_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        v_stream_TREADY = grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_v_stream_TREADY;
    end else begin
        v_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((icmp_ln220_fu_534_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((grp_quantize_v_group_fu_422_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state5))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln220_fu_540_p2 = (indvar_flatten_fu_92 + 11'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start = grp_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3_fu_352_ap_start_reg;

assign grp_quantize_v_group_fu_422_ap_start = grp_quantize_v_group_fu_422_ap_start_reg;

assign grp_quantize_v_group_fu_422_vq_cache_o_stream_TREADY = (vq_cache_o_stream_TREADY & ap_CS_fsm_state5);

assign grp_quantize_v_group_fu_422_vs_cache_o_stream_TREADY = (vs_cache_o_stream_TREADY & ap_CS_fsm_state5);

assign hct_2_fu_854_p2 = (select_ln220_reg_1272 + 4'd1);

assign icmp_ln220_fu_534_p2 = ((indvar_flatten_fu_92 == 11'd1024) ? 1'b1 : 1'b0);

assign icmp_ln221_fu_563_p2 = ((hct_fu_84 == 4'd8) ? 1'b1 : 1'b0);

assign select_ln220_1_fu_577_p3 = ((icmp_ln221_fu_563_p2[0:0] == 1'b1) ? tt_2_fu_557_p2 : tt_fu_88);

assign select_ln220_fu_569_p3 = ((icmp_ln221_fu_563_p2[0:0] == 1'b1) ? 4'd0 : hct_fu_84);

assign trunc_ln221_fu_585_p1 = select_ln220_fu_569_p3[2:0];

assign tt_2_fu_557_p2 = (tt_fu_88 + 8'd1);

assign vq_buf_0_address1 = grp_quantize_v_group_fu_422_vq_buf_0_address1;

assign vq_buf_0_ce1 = grp_quantize_v_group_fu_422_vq_buf_0_ce1;

assign vq_buf_0_d1 = grp_quantize_v_group_fu_422_vq_buf_0_d1;

assign vq_buf_0_we1 = grp_quantize_v_group_fu_422_vq_buf_0_we1;

assign vq_buf_1_address1 = grp_quantize_v_group_fu_422_vq_buf_1_address1;

assign vq_buf_1_ce1 = grp_quantize_v_group_fu_422_vq_buf_1_ce1;

assign vq_buf_1_d1 = grp_quantize_v_group_fu_422_vq_buf_1_d1;

assign vq_buf_1_we1 = grp_quantize_v_group_fu_422_vq_buf_1_we1;

assign vq_buf_2_address1 = grp_quantize_v_group_fu_422_vq_buf_2_address1;

assign vq_buf_2_ce1 = grp_quantize_v_group_fu_422_vq_buf_2_ce1;

assign vq_buf_2_d1 = grp_quantize_v_group_fu_422_vq_buf_2_d1;

assign vq_buf_2_we1 = grp_quantize_v_group_fu_422_vq_buf_2_we1;

assign vq_buf_3_address1 = grp_quantize_v_group_fu_422_vq_buf_3_address1;

assign vq_buf_3_ce1 = grp_quantize_v_group_fu_422_vq_buf_3_ce1;

assign vq_buf_3_d1 = grp_quantize_v_group_fu_422_vq_buf_3_d1;

assign vq_buf_3_we1 = grp_quantize_v_group_fu_422_vq_buf_3_we1;

assign vq_buf_4_address1 = grp_quantize_v_group_fu_422_vq_buf_4_address1;

assign vq_buf_4_ce1 = grp_quantize_v_group_fu_422_vq_buf_4_ce1;

assign vq_buf_4_d1 = grp_quantize_v_group_fu_422_vq_buf_4_d1;

assign vq_buf_4_we1 = grp_quantize_v_group_fu_422_vq_buf_4_we1;

assign vq_buf_5_address1 = grp_quantize_v_group_fu_422_vq_buf_5_address1;

assign vq_buf_5_ce1 = grp_quantize_v_group_fu_422_vq_buf_5_ce1;

assign vq_buf_5_d1 = grp_quantize_v_group_fu_422_vq_buf_5_d1;

assign vq_buf_5_we1 = grp_quantize_v_group_fu_422_vq_buf_5_we1;

assign vq_buf_6_address1 = grp_quantize_v_group_fu_422_vq_buf_6_address1;

assign vq_buf_6_ce1 = grp_quantize_v_group_fu_422_vq_buf_6_ce1;

assign vq_buf_6_d1 = grp_quantize_v_group_fu_422_vq_buf_6_d1;

assign vq_buf_6_we1 = grp_quantize_v_group_fu_422_vq_buf_6_we1;

assign vq_buf_7_address1 = grp_quantize_v_group_fu_422_vq_buf_7_address1;

assign vq_buf_7_ce1 = grp_quantize_v_group_fu_422_vq_buf_7_ce1;

assign vq_buf_7_d1 = grp_quantize_v_group_fu_422_vq_buf_7_d1;

assign vq_buf_7_we1 = grp_quantize_v_group_fu_422_vq_buf_7_we1;

assign vq_cache_o_stream_TDATA = grp_quantize_v_group_fu_422_vq_cache_o_stream_TDATA;

assign vq_cache_o_stream_TVALID = grp_quantize_v_group_fu_422_vq_cache_o_stream_TVALID;

assign vs_buf_address0 = grp_quantize_v_group_fu_422_vs_buf_address0;

assign vs_buf_address1 = grp_quantize_v_group_fu_422_vs_buf_address1;

assign vs_buf_ce0 = grp_quantize_v_group_fu_422_vs_buf_ce0;

assign vs_buf_ce1 = grp_quantize_v_group_fu_422_vs_buf_ce1;

assign vs_buf_d1 = grp_quantize_v_group_fu_422_vs_buf_d1;

assign vs_buf_we1 = grp_quantize_v_group_fu_422_vs_buf_we1;

assign vs_cache_o_stream_TDATA = grp_quantize_v_group_fu_422_vs_cache_o_stream_TDATA;

assign vs_cache_o_stream_TVALID = grp_quantize_v_group_fu_422_vs_cache_o_stream_TVALID;

assign zext_ln220_fu_594_p1 = select_ln220_1_reg_1277;

always @ (posedge ap_clk) begin
    zext_ln220_reg_1287[9:8] <= 2'b00;
end

endmodule //RV_GEMM_load_quantize_vit_v
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
`timescale 1ns / 1ps

module RV_GEMM_sparsemux_17_3_23_1_1 (din0,din1,din2,din3,din4,din5,din6,din7,def,sel,dout);

parameter din0_WIDTH = 1;

parameter din1_WIDTH = 1;

parameter din2_WIDTH = 1;

parameter din3_WIDTH = 1;

parameter din4_WIDTH = 1;

parameter din5_WIDTH = 1;

parameter din6_WIDTH = 1;

parameter din7_WIDTH = 1;

parameter def_WIDTH = 1;
parameter sel_WIDTH = 1;
parameter dout_WIDTH = 1;

parameter [sel_WIDTH-1:0] CASE0 = 1;

parameter [sel_WIDTH-1:0] CASE1 = 1;

parameter [sel_WIDTH-1:0] CASE2 = 1;

parameter [sel_WIDTH-1:0] CASE3 = 1;

parameter [sel_WIDTH-1:0] CASE4 = 1;

parameter [sel_WIDTH-1:0] CASE5 = 1;

parameter [sel_WIDTH-1:0] CASE6 = 1;

parameter [sel_WIDTH-1:0] CASE7 = 1;

parameter ID = 1;
parameter NUM_STAGE = 1;



input [din0_WIDTH-1:0] din0;

input [din1_WIDTH-1:0] din1;

input [din2_WIDTH-1:0] din2;

input [din3_WIDTH-1:0] din3;

input [din4_WIDTH-1:0] din4;

input [din5_WIDTH-1:0] din5;

input [din6_WIDTH-1:0] din6;

input [din7_WIDTH-1:0] din7;

input [def_WIDTH-1:0] def;
input [sel_WIDTH-1:0] sel;

output [dout_WIDTH-1:0] dout;



reg [dout_WIDTH-1:0] dout_tmp;

always @ (*) begin
case (sel)
    
    CASE0 : dout_tmp = din0;
    
    CASE1 : dout_tmp = din1;
    
    CASE2 : dout_tmp = din2;
    
    CASE3 : dout_tmp = din3;
    
    CASE4 : dout_tmp = din4;
    
    CASE5 : dout_tmp = din5;
    
    CASE6 : dout_tmp = din6;
    
    CASE7 : dout_tmp = din7;
    
    default : dout_tmp = def;
endcase
end


assign dout = dout_tmp;



endmodule
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module RV_GEMM_rv_gemm_one_layer_vq_buf_RAM_2P_URAM_2R1W (
     
    address0, ce0,
    
    q0, 
     
    address1, ce1,
    d1, we1, 
    
    
    reset, clk);

parameter DataWidth = 64;
parameter AddressWidth = 10;
parameter AddressRange = 1024;
parameter COL_WIDTH = 8;
parameter NUM_COL = (DataWidth/COL_WIDTH);

input[AddressWidth-1:0] address0;
input ce0;

output wire[DataWidth-1:0] q0; 

input[AddressWidth-1:0] address1;
input ce1;
input[DataWidth-1:0] d1;
input [NUM_COL-1:0] we1; 


input reset;
input clk;

(* ram_style = "hls_ultra" , cascade_height = 1 *)reg [DataWidth-1:0] ram[0:AddressRange-1];

reg [DataWidth-1:0] q0_t0;
(* retiming_backward = 1 *)reg [DataWidth-1:0] q0_t1; 
reg [DataWidth-1:0] q1_t0;
(* retiming_backward = 1 *)reg [DataWidth-1:0] q1_t1; 



assign q0 = q0_t1;




always @(posedge clk)  
begin

 
    if (ce0) 
    begin 
        q0_t1 <= q0_t0;
    end   
  
end 


genvar i;





always @(posedge clk) begin 
    if (ce0) begin
        q0_t0 <= ram[address0];
    end
end 

 
 

generate
    for (i=0;i<NUM_COL;i=i+1) begin
        always @(posedge clk) begin
            if (ce1) begin
                if (we1[i])
                    ram[address1][i*COL_WIDTH +: COL_WIDTH] <= d1[i*COL_WIDTH +: COL_WIDTH]; 
            end
        end
    end
endgenerate




 
 

endmodule

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module RV_GEMM_rv_gemm_one_layer_vs_buf_RAM_2P_BRAM_2R1W (
      
    address0, ce0,
    
    q0, 
      
    address1, ce1,
    d1, we1, 
    
    
    reset, clk);

parameter DataWidth = 32;
parameter AddressWidth = 10;
parameter AddressRange = 1024;
 
input[AddressWidth-1:0] address0;
input ce0;

output wire[DataWidth-1:0] q0; 
 
input[AddressWidth-1:0] address1;
input ce1;
input[DataWidth-1:0] d1;
input we1; 


input reset;
input clk;

(* ram_style = "block"  *)reg [DataWidth-1:0] ram[0:AddressRange-1];
 
reg [DataWidth-1:0] q0_t0;
reg [DataWidth-1:0] q0_t1;  
reg [DataWidth-1:0] q1_t0;
reg [DataWidth-1:0] q1_t1; 



 
assign q0 = q0_t1;
 



always @(posedge clk)  
begin
 
 
    if (ce0) 
    begin
        q0_t1 <= q0_t0;
    end   
 
end 

 



always @(posedge clk) 
begin 
    if (ce0) begin
        q0_t0 <= ram[address0];
    end
end 

 
 

always @(posedge clk)  
begin 
    if (ce1) begin
        if (we1) 
            ram[address1] <= d1; 
    end
end 



 


endmodule

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W (
     
    address0, ce0,
    
    q0, 
      
    address1, ce1,
    d1, we1, 
    
     
    reset, clk);

parameter DataWidth = 8;
parameter AddressWidth = 7;
parameter AddressRange = 128;
 
input[AddressWidth-1:0] address0;
input ce0;

output reg[DataWidth-1:0] q0; 
 
input[AddressWidth-1:0] address1;
input ce1;
input[DataWidth-1:0] d1;
input we1; 


input reset;
input clk;

(* ram_style = "distributed"  *)reg [DataWidth-1:0] ram[0:AddressRange-1];


 



always @(posedge clk) 
begin 
    if (ce0) begin
        q0 <= ram[address0];
    end
end 

 
  

always @(posedge clk)  
begin 
    if (ce1) begin
        if (we1) 
            ram[address1] <= d1; 
    end
end 



 
 

endmodule

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="RV_GEMM_RV_GEMM,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xck26-sfvc784-2LV-c,HLS_INPUT_CLOCK=2.500000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=2.136500,HLS_SYN_LAT=59868994,HLS_SYN_TPT=none,HLS_SYN_MEM=2,HLS_SYN_DSP=0,HLS_SYN_FF=15586,HLS_SYN_LUT=22202,HLS_VERSION=2023_2}" *)

module RV_GEMM (
        ap_clk,
        ap_rst_n,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        mode,
        l_begin,
        l_close,
        pos_r,
        rq_stream_TDATA,
        rq_stream_TVALID,
        rq_stream_TREADY,
        rs_stream_TDATA,
        rs_stream_TVALID,
        rs_stream_TREADY,
        v_stream_TDATA,
        v_stream_TVALID,
        v_stream_TREADY,
        vq_cache_i_stream_TDATA,
        vq_cache_i_stream_TVALID,
        vq_cache_i_stream_TREADY,
        vs_cache_i_stream_TDATA,
        vs_cache_i_stream_TVALID,
        vs_cache_i_stream_TREADY,
        vq_cache_o_stream_TDATA,
        vq_cache_o_stream_TVALID,
        vq_cache_o_stream_TREADY,
        vs_cache_o_stream_TDATA,
        vs_cache_o_stream_TVALID,
        vs_cache_o_stream_TREADY,
        aq_stream_TDATA,
        aq_stream_TVALID,
        aq_stream_TREADY,
        as_stream_TDATA,
        as_stream_TVALID,
        as_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 5'd1;
parameter    ap_ST_fsm_state2 = 5'd2;
parameter    ap_ST_fsm_state3 = 5'd4;
parameter    ap_ST_fsm_state4 = 5'd8;
parameter    ap_ST_fsm_state5 = 5'd16;

input   ap_clk;
input   ap_rst_n;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [0:0] mode;
input  [31:0] l_begin;
input  [31:0] l_close;
input  [31:0] pos_r;
input  [63:0] rq_stream_TDATA;
input   rq_stream_TVALID;
output   rq_stream_TREADY;
input  [7:0] rs_stream_TDATA;
input   rs_stream_TVALID;
output   rs_stream_TREADY;
input  [183:0] v_stream_TDATA;
input   v_stream_TVALID;
output   v_stream_TREADY;
input  [63:0] vq_cache_i_stream_TDATA;
input   vq_cache_i_stream_TVALID;
output   vq_cache_i_stream_TREADY;
input  [7:0] vs_cache_i_stream_TDATA;
input   vs_cache_i_stream_TVALID;
output   vs_cache_i_stream_TREADY;
output  [63:0] vq_cache_o_stream_TDATA;
output   vq_cache_o_stream_TVALID;
input   vq_cache_o_stream_TREADY;
output  [7:0] vs_cache_o_stream_TDATA;
output   vs_cache_o_stream_TVALID;
input   vs_cache_o_stream_TREADY;
output  [63:0] aq_stream_TDATA;
output   aq_stream_TVALID;
input   aq_stream_TREADY;
output  [7:0] as_stream_TDATA;
output   as_stream_TVALID;
input   as_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [4:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_block_state1;
wire   [31:0] select_ln460_fu_130_p3;
reg   [31:0] select_ln460_reg_208;
wire   [0:0] and_ln59_fu_170_p2;
reg   [0:0] and_ln59_reg_219;
wire    ap_CS_fsm_state2;
wire    grp_rv_gemm_one_layer_fu_106_ap_start;
wire    grp_rv_gemm_one_layer_fu_106_ap_done;
wire    grp_rv_gemm_one_layer_fu_106_ap_idle;
wire    grp_rv_gemm_one_layer_fu_106_ap_ready;
wire    grp_rv_gemm_one_layer_fu_106_rq_stream_TREADY;
wire    grp_rv_gemm_one_layer_fu_106_rs_stream_TREADY;
wire    grp_rv_gemm_one_layer_fu_106_v_stream_TREADY;
wire    grp_rv_gemm_one_layer_fu_106_vq_cache_i_stream_TREADY;
wire    grp_rv_gemm_one_layer_fu_106_vs_cache_i_stream_TREADY;
wire   [63:0] grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TDATA;
wire    grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TVALID;
wire    grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TREADY;
wire   [7:0] grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TDATA;
wire    grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TVALID;
wire    grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TREADY;
wire   [63:0] grp_rv_gemm_one_layer_fu_106_aq_stream_TDATA;
wire    grp_rv_gemm_one_layer_fu_106_aq_stream_TVALID;
wire    grp_rv_gemm_one_layer_fu_106_aq_stream_TREADY;
wire   [7:0] grp_rv_gemm_one_layer_fu_106_as_stream_TDATA;
wire    grp_rv_gemm_one_layer_fu_106_as_stream_TVALID;
wire    grp_rv_gemm_one_layer_fu_106_as_stream_TREADY;
reg    grp_rv_gemm_one_layer_fu_106_ap_start_reg;
wire    ap_CS_fsm_state3;
wire    ap_CS_fsm_state4;
reg   [31:0] l_1_fu_78;
wire   [31:0] l_fu_176_p2;
reg    ap_block_state4_on_subcall_done;
wire   [0:0] tmp_fu_151_p3;
wire   [0:0] icmp_ln59_fu_165_p2;
wire   [0:0] xor_ln59_fu_159_p2;
wire   [0:0] icmp_ln460_fu_146_p2;
wire    ap_CS_fsm_state5;
wire    regslice_both_vq_cache_o_stream_U_apdone_blk;
wire    regslice_both_vs_cache_o_stream_U_apdone_blk;
wire    regslice_both_aq_stream_U_apdone_blk;
wire    regslice_both_as_stream_U_apdone_blk;
reg    ap_block_state5;
reg   [4:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
wire    regslice_both_rq_stream_U_apdone_blk;
wire   [63:0] rq_stream_TDATA_int_regslice;
wire    rq_stream_TVALID_int_regslice;
reg    rq_stream_TREADY_int_regslice;
wire    regslice_both_rq_stream_U_ack_in;
wire    regslice_both_rs_stream_U_apdone_blk;
wire   [7:0] rs_stream_TDATA_int_regslice;
wire    rs_stream_TVALID_int_regslice;
reg    rs_stream_TREADY_int_regslice;
wire    regslice_both_rs_stream_U_ack_in;
wire    regslice_both_v_stream_U_apdone_blk;
wire   [183:0] v_stream_TDATA_int_regslice;
wire    v_stream_TVALID_int_regslice;
reg    v_stream_TREADY_int_regslice;
wire    regslice_both_v_stream_U_ack_in;
wire    regslice_both_vq_cache_i_stream_U_apdone_blk;
wire   [63:0] vq_cache_i_stream_TDATA_int_regslice;
wire    vq_cache_i_stream_TVALID_int_regslice;
reg    vq_cache_i_stream_TREADY_int_regslice;
wire    regslice_both_vq_cache_i_stream_U_ack_in;
wire    regslice_both_vs_cache_i_stream_U_apdone_blk;
wire   [7:0] vs_cache_i_stream_TDATA_int_regslice;
wire    vs_cache_i_stream_TVALID_int_regslice;
reg    vs_cache_i_stream_TREADY_int_regslice;
wire    regslice_both_vs_cache_i_stream_U_ack_in;
wire    vq_cache_o_stream_TREADY_int_regslice;
wire    regslice_both_vq_cache_o_stream_U_vld_out;
wire    vs_cache_o_stream_TREADY_int_regslice;
wire    regslice_both_vs_cache_o_stream_U_vld_out;
wire    aq_stream_TREADY_int_regslice;
wire    regslice_both_aq_stream_U_vld_out;
wire    as_stream_TREADY_int_regslice;
wire    regslice_both_as_stream_U_vld_out;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 5'd1;
//#0 grp_rv_gemm_one_layer_fu_106_ap_start_reg = 1'b0;
//#0 l_1_fu_78 = 32'd0;
end

RV_GEMM_rv_gemm_one_layer grp_rv_gemm_one_layer_fu_106(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_rv_gemm_one_layer_fu_106_ap_start),
    .ap_done(grp_rv_gemm_one_layer_fu_106_ap_done),
    .ap_idle(grp_rv_gemm_one_layer_fu_106_ap_idle),
    .ap_ready(grp_rv_gemm_one_layer_fu_106_ap_ready),
    .mode_val(mode),
    .pos_r(pos_r),
    .rq_stream_TDATA(rq_stream_TDATA_int_regslice),
    .rq_stream_TVALID(rq_stream_TVALID_int_regslice),
    .rq_stream_TREADY(grp_rv_gemm_one_layer_fu_106_rq_stream_TREADY),
    .rs_stream_TDATA(rs_stream_TDATA_int_regslice),
    .rs_stream_TVALID(rs_stream_TVALID_int_regslice),
    .rs_stream_TREADY(grp_rv_gemm_one_layer_fu_106_rs_stream_TREADY),
    .v_stream_TDATA(v_stream_TDATA_int_regslice),
    .v_stream_TVALID(v_stream_TVALID_int_regslice),
    .v_stream_TREADY(grp_rv_gemm_one_layer_fu_106_v_stream_TREADY),
    .vq_cache_i_stream_TDATA(vq_cache_i_stream_TDATA_int_regslice),
    .vq_cache_i_stream_TVALID(vq_cache_i_stream_TVALID_int_regslice),
    .vq_cache_i_stream_TREADY(grp_rv_gemm_one_layer_fu_106_vq_cache_i_stream_TREADY),
    .vs_cache_i_stream_TDATA(vs_cache_i_stream_TDATA_int_regslice),
    .vs_cache_i_stream_TVALID(vs_cache_i_stream_TVALID_int_regslice),
    .vs_cache_i_stream_TREADY(grp_rv_gemm_one_layer_fu_106_vs_cache_i_stream_TREADY),
    .vq_cache_o_stream_TDATA(grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TDATA),
    .vq_cache_o_stream_TVALID(grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TVALID),
    .vq_cache_o_stream_TREADY(grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TREADY),
    .vs_cache_o_stream_TDATA(grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TDATA),
    .vs_cache_o_stream_TVALID(grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TVALID),
    .vs_cache_o_stream_TREADY(grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TREADY),
    .aq_stream_TDATA(grp_rv_gemm_one_layer_fu_106_aq_stream_TDATA),
    .aq_stream_TVALID(grp_rv_gemm_one_layer_fu_106_aq_stream_TVALID),
    .aq_stream_TREADY(grp_rv_gemm_one_layer_fu_106_aq_stream_TREADY),
    .as_stream_TDATA(grp_rv_gemm_one_layer_fu_106_as_stream_TDATA),
    .as_stream_TVALID(grp_rv_gemm_one_layer_fu_106_as_stream_TVALID),
    .as_stream_TREADY(grp_rv_gemm_one_layer_fu_106_as_stream_TREADY)
);

RV_GEMM_regslice_both #(
    .DataWidth( 64 ))
regslice_both_rq_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(rq_stream_TDATA),
    .vld_in(rq_stream_TVALID),
    .ack_in(regslice_both_rq_stream_U_ack_in),
    .data_out(rq_stream_TDATA_int_regslice),
    .vld_out(rq_stream_TVALID_int_regslice),
    .ack_out(rq_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_rq_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 8 ))
regslice_both_rs_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(rs_stream_TDATA),
    .vld_in(rs_stream_TVALID),
    .ack_in(regslice_both_rs_stream_U_ack_in),
    .data_out(rs_stream_TDATA_int_regslice),
    .vld_out(rs_stream_TVALID_int_regslice),
    .ack_out(rs_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_rs_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 184 ))
regslice_both_v_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(v_stream_TDATA),
    .vld_in(v_stream_TVALID),
    .ack_in(regslice_both_v_stream_U_ack_in),
    .data_out(v_stream_TDATA_int_regslice),
    .vld_out(v_stream_TVALID_int_regslice),
    .ack_out(v_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_v_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 64 ))
regslice_both_vq_cache_i_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(vq_cache_i_stream_TDATA),
    .vld_in(vq_cache_i_stream_TVALID),
    .ack_in(regslice_both_vq_cache_i_stream_U_ack_in),
    .data_out(vq_cache_i_stream_TDATA_int_regslice),
    .vld_out(vq_cache_i_stream_TVALID_int_regslice),
    .ack_out(vq_cache_i_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_vq_cache_i_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 8 ))
regslice_both_vs_cache_i_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(vs_cache_i_stream_TDATA),
    .vld_in(vs_cache_i_stream_TVALID),
    .ack_in(regslice_both_vs_cache_i_stream_U_ack_in),
    .data_out(vs_cache_i_stream_TDATA_int_regslice),
    .vld_out(vs_cache_i_stream_TVALID_int_regslice),
    .ack_out(vs_cache_i_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_vs_cache_i_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 64 ))
regslice_both_vq_cache_o_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TDATA),
    .vld_in(grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TVALID),
    .ack_in(vq_cache_o_stream_TREADY_int_regslice),
    .data_out(vq_cache_o_stream_TDATA),
    .vld_out(regslice_both_vq_cache_o_stream_U_vld_out),
    .ack_out(vq_cache_o_stream_TREADY),
    .apdone_blk(regslice_both_vq_cache_o_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 8 ))
regslice_both_vs_cache_o_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TDATA),
    .vld_in(grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TVALID),
    .ack_in(vs_cache_o_stream_TREADY_int_regslice),
    .data_out(vs_cache_o_stream_TDATA),
    .vld_out(regslice_both_vs_cache_o_stream_U_vld_out),
    .ack_out(vs_cache_o_stream_TREADY),
    .apdone_blk(regslice_both_vs_cache_o_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 64 ))
regslice_both_aq_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_rv_gemm_one_layer_fu_106_aq_stream_TDATA),
    .vld_in(grp_rv_gemm_one_layer_fu_106_aq_stream_TVALID),
    .ack_in(aq_stream_TREADY_int_regslice),
    .data_out(aq_stream_TDATA),
    .vld_out(regslice_both_aq_stream_U_vld_out),
    .ack_out(aq_stream_TREADY),
    .apdone_blk(regslice_both_aq_stream_U_apdone_blk)
);

RV_GEMM_regslice_both #(
    .DataWidth( 8 ))
regslice_both_as_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_rv_gemm_one_layer_fu_106_as_stream_TDATA),
    .vld_in(grp_rv_gemm_one_layer_fu_106_as_stream_TVALID),
    .ack_in(as_stream_TREADY_int_regslice),
    .data_out(as_stream_TDATA),
    .vld_out(regslice_both_as_stream_U_vld_out),
    .ack_out(as_stream_TREADY),
    .apdone_blk(regslice_both_as_stream_U_apdone_blk)
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
        end else if (((1'b0 == ap_block_state5) & (1'b1 == ap_CS_fsm_state5))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_rv_gemm_one_layer_fu_106_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state3)) begin
            grp_rv_gemm_one_layer_fu_106_ap_start_reg <= 1'b1;
        end else if ((grp_rv_gemm_one_layer_fu_106_ap_ready == 1'b1)) begin
            grp_rv_gemm_one_layer_fu_106_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        l_1_fu_78 <= l_begin;
    end else if (((1'b0 == ap_block_state4_on_subcall_done) & (1'b1 == ap_CS_fsm_state4))) begin
        l_1_fu_78 <= l_fu_176_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        and_ln59_reg_219 <= and_ln59_fu_170_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        select_ln460_reg_208[3 : 2] <= select_ln460_fu_130_p3[3 : 2];
select_ln460_reg_208[5] <= select_ln460_fu_130_p3[5];
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

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_on_subcall_done)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state5)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state5) & (1'b1 == ap_CS_fsm_state5))) begin
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
    if (((1'b0 == ap_block_state5) & (1'b1 == ap_CS_fsm_state5))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) & (1'd1 == and_ln59_reg_219))) begin
        rq_stream_TREADY_int_regslice = grp_rv_gemm_one_layer_fu_106_rq_stream_TREADY;
    end else begin
        rq_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) & (1'd1 == and_ln59_reg_219))) begin
        rs_stream_TREADY_int_regslice = grp_rv_gemm_one_layer_fu_106_rs_stream_TREADY;
    end else begin
        rs_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) & (1'd1 == and_ln59_reg_219))) begin
        v_stream_TREADY_int_regslice = grp_rv_gemm_one_layer_fu_106_v_stream_TREADY;
    end else begin
        v_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) & (1'd1 == and_ln59_reg_219))) begin
        vq_cache_i_stream_TREADY_int_regslice = grp_rv_gemm_one_layer_fu_106_vq_cache_i_stream_TREADY;
    end else begin
        vq_cache_i_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) & (1'd1 == and_ln59_reg_219))) begin
        vs_cache_i_stream_TREADY_int_regslice = grp_rv_gemm_one_layer_fu_106_vs_cache_i_stream_TREADY;
    end else begin
        vs_cache_i_stream_TREADY_int_regslice = 1'b0;
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
            if (((1'd0 == and_ln59_fu_170_p2) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln460_fu_146_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln460_fu_146_p2 == 1'd1) & (1'd1 == and_ln59_fu_170_p2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            if (((1'b0 == ap_block_state4_on_subcall_done) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            if (((1'b0 == ap_block_state5) & (1'b1 == ap_CS_fsm_state5))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign and_ln59_fu_170_p2 = (xor_ln59_fu_159_p2 & icmp_ln59_fu_165_p2);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

always @ (*) begin
    ap_block_state1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0));
end

always @ (*) begin
    ap_block_state4_on_subcall_done = ((grp_rv_gemm_one_layer_fu_106_ap_done == 1'b0) & (1'd1 == and_ln59_reg_219));
end

always @ (*) begin
    ap_block_state5 = ((regslice_both_as_stream_U_apdone_blk == 1'b1) | (regslice_both_aq_stream_U_apdone_blk == 1'b1) | (regslice_both_vs_cache_o_stream_U_apdone_blk == 1'b1) | (regslice_both_vq_cache_o_stream_U_apdone_blk == 1'b1));
end

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign aq_stream_TVALID = regslice_both_aq_stream_U_vld_out;

assign as_stream_TVALID = regslice_both_as_stream_U_vld_out;

assign grp_rv_gemm_one_layer_fu_106_ap_start = grp_rv_gemm_one_layer_fu_106_ap_start_reg;

assign grp_rv_gemm_one_layer_fu_106_aq_stream_TREADY = (aq_stream_TREADY_int_regslice & ap_CS_fsm_state4);

assign grp_rv_gemm_one_layer_fu_106_as_stream_TREADY = (as_stream_TREADY_int_regslice & ap_CS_fsm_state4);

assign grp_rv_gemm_one_layer_fu_106_vq_cache_o_stream_TREADY = (vq_cache_o_stream_TREADY_int_regslice & ap_CS_fsm_state4);

assign grp_rv_gemm_one_layer_fu_106_vs_cache_o_stream_TREADY = (vs_cache_o_stream_TREADY_int_regslice & ap_CS_fsm_state4);

assign icmp_ln460_fu_146_p2 = (($signed(l_1_fu_78) < $signed(l_close)) ? 1'b1 : 1'b0);

assign icmp_ln59_fu_165_p2 = (($signed(l_1_fu_78) < $signed(select_ln460_reg_208)) ? 1'b1 : 1'b0);

assign l_fu_176_p2 = (l_1_fu_78 + 32'd1);

assign rq_stream_TREADY = regslice_both_rq_stream_U_ack_in;

assign rs_stream_TREADY = regslice_both_rs_stream_U_ack_in;

assign select_ln460_fu_130_p3 = ((mode[0:0] == 1'b1) ? 32'd12 : 32'd32);

assign tmp_fu_151_p3 = l_1_fu_78[32'd31];

assign v_stream_TREADY = regslice_both_v_stream_U_ack_in;

assign vq_cache_i_stream_TREADY = regslice_both_vq_cache_i_stream_U_ack_in;

assign vq_cache_o_stream_TVALID = regslice_both_vq_cache_o_stream_U_vld_out;

assign vs_cache_i_stream_TREADY = regslice_both_vs_cache_i_stream_U_ack_in;

assign vs_cache_o_stream_TVALID = regslice_both_vs_cache_o_stream_U_vld_out;

assign xor_ln59_fu_159_p2 = (tmp_fu_151_p3 ^ 1'd1);

always @ (posedge ap_clk) begin
    select_ln460_reg_208[1:0] <= 2'b00;
    select_ln460_reg_208[4:4] <= 1'b0;
    select_ln460_reg_208[31:6] <= 26'b00000000000000000000000000;
end

endmodule //RV_GEMM
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_quantize_v_group (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        vq_cache_o_stream_TREADY,
        vs_cache_o_stream_TREADY,
        hct,
        st,
        raw_tile_0_0_val,
        raw_tile_0_1_val,
        raw_tile_0_2_val,
        raw_tile_0_3_val,
        raw_tile_0_4_val,
        raw_tile_0_5_val,
        raw_tile_0_6_val,
        raw_tile_0_7_val,
        raw_tile_1_0_val,
        raw_tile_1_1_val,
        raw_tile_1_2_val,
        raw_tile_1_3_val,
        raw_tile_1_4_val,
        raw_tile_1_5_val,
        raw_tile_1_6_val,
        raw_tile_1_7_val,
        raw_tile_2_0_val,
        raw_tile_2_1_val,
        raw_tile_2_2_val,
        raw_tile_2_3_val,
        raw_tile_2_4_val,
        raw_tile_2_5_val,
        raw_tile_2_6_val,
        raw_tile_2_7_val,
        raw_tile_3_0_val,
        raw_tile_3_1_val,
        raw_tile_3_2_val,
        raw_tile_3_3_val,
        raw_tile_3_4_val,
        raw_tile_3_5_val,
        raw_tile_3_6_val,
        raw_tile_3_7_val,
        raw_tile_4_0_val,
        raw_tile_4_1_val,
        raw_tile_4_2_val,
        raw_tile_4_3_val,
        raw_tile_4_4_val,
        raw_tile_4_5_val,
        raw_tile_4_6_val,
        raw_tile_4_7_val,
        raw_tile_5_0_val,
        raw_tile_5_1_val,
        raw_tile_5_2_val,
        raw_tile_5_3_val,
        raw_tile_5_4_val,
        raw_tile_5_5_val,
        raw_tile_5_6_val,
        raw_tile_5_7_val,
        raw_tile_6_0_val,
        raw_tile_6_1_val,
        raw_tile_6_2_val,
        raw_tile_6_3_val,
        raw_tile_6_4_val,
        raw_tile_6_5_val,
        raw_tile_6_6_val,
        raw_tile_6_7_val,
        raw_tile_7_0_val,
        raw_tile_7_1_val,
        raw_tile_7_2_val,
        raw_tile_7_3_val,
        raw_tile_7_4_val,
        raw_tile_7_5_val,
        raw_tile_7_6_val,
        raw_tile_7_7_val,
        write_cache_stream,
        vq_buf_0_address1,
        vq_buf_0_ce1,
        vq_buf_0_we1,
        vq_buf_0_d1,
        vq_buf_1_address1,
        vq_buf_1_ce1,
        vq_buf_1_we1,
        vq_buf_1_d1,
        vq_buf_2_address1,
        vq_buf_2_ce1,
        vq_buf_2_we1,
        vq_buf_2_d1,
        vq_buf_3_address1,
        vq_buf_3_ce1,
        vq_buf_3_we1,
        vq_buf_3_d1,
        vq_buf_4_address1,
        vq_buf_4_ce1,
        vq_buf_4_we1,
        vq_buf_4_d1,
        vq_buf_5_address1,
        vq_buf_5_ce1,
        vq_buf_5_we1,
        vq_buf_5_d1,
        vq_buf_6_address1,
        vq_buf_6_ce1,
        vq_buf_6_we1,
        vq_buf_6_d1,
        vq_buf_7_address1,
        vq_buf_7_ce1,
        vq_buf_7_we1,
        vq_buf_7_d1,
        vs_buf_address0,
        vs_buf_ce0,
        vs_buf_q0,
        vs_buf_address1,
        vs_buf_ce1,
        vs_buf_we1,
        vs_buf_d1,
        vq_cache_o_stream_TDATA,
        vq_cache_o_stream_TVALID,
        vs_cache_o_stream_TDATA,
        vs_cache_o_stream_TVALID
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
parameter    ap_ST_iter7_fsm_state15 = 3'd2;
parameter    ap_ST_iter7_fsm_state16 = 3'd4;
parameter    ap_ST_iter8_fsm_state17 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 3'd1;
parameter    ap_ST_iter2_fsm_state0 = 3'd1;
parameter    ap_ST_iter3_fsm_state0 = 3'd1;
parameter    ap_ST_iter4_fsm_state0 = 3'd1;
parameter    ap_ST_iter5_fsm_state0 = 3'd1;
parameter    ap_ST_iter6_fsm_state0 = 3'd1;
parameter    ap_ST_iter7_fsm_state0 = 3'd1;
parameter    ap_ST_iter8_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   vq_cache_o_stream_TREADY;
input   vs_cache_o_stream_TREADY;
input  [2:0] hct;
input  [9:0] st;
input  [22:0] raw_tile_0_0_val;
input  [22:0] raw_tile_0_1_val;
input  [22:0] raw_tile_0_2_val;
input  [22:0] raw_tile_0_3_val;
input  [22:0] raw_tile_0_4_val;
input  [22:0] raw_tile_0_5_val;
input  [22:0] raw_tile_0_6_val;
input  [22:0] raw_tile_0_7_val;
input  [22:0] raw_tile_1_0_val;
input  [22:0] raw_tile_1_1_val;
input  [22:0] raw_tile_1_2_val;
input  [22:0] raw_tile_1_3_val;
input  [22:0] raw_tile_1_4_val;
input  [22:0] raw_tile_1_5_val;
input  [22:0] raw_tile_1_6_val;
input  [22:0] raw_tile_1_7_val;
input  [22:0] raw_tile_2_0_val;
input  [22:0] raw_tile_2_1_val;
input  [22:0] raw_tile_2_2_val;
input  [22:0] raw_tile_2_3_val;
input  [22:0] raw_tile_2_4_val;
input  [22:0] raw_tile_2_5_val;
input  [22:0] raw_tile_2_6_val;
input  [22:0] raw_tile_2_7_val;
input  [22:0] raw_tile_3_0_val;
input  [22:0] raw_tile_3_1_val;
input  [22:0] raw_tile_3_2_val;
input  [22:0] raw_tile_3_3_val;
input  [22:0] raw_tile_3_4_val;
input  [22:0] raw_tile_3_5_val;
input  [22:0] raw_tile_3_6_val;
input  [22:0] raw_tile_3_7_val;
input  [22:0] raw_tile_4_0_val;
input  [22:0] raw_tile_4_1_val;
input  [22:0] raw_tile_4_2_val;
input  [22:0] raw_tile_4_3_val;
input  [22:0] raw_tile_4_4_val;
input  [22:0] raw_tile_4_5_val;
input  [22:0] raw_tile_4_6_val;
input  [22:0] raw_tile_4_7_val;
input  [22:0] raw_tile_5_0_val;
input  [22:0] raw_tile_5_1_val;
input  [22:0] raw_tile_5_2_val;
input  [22:0] raw_tile_5_3_val;
input  [22:0] raw_tile_5_4_val;
input  [22:0] raw_tile_5_5_val;
input  [22:0] raw_tile_5_6_val;
input  [22:0] raw_tile_5_7_val;
input  [22:0] raw_tile_6_0_val;
input  [22:0] raw_tile_6_1_val;
input  [22:0] raw_tile_6_2_val;
input  [22:0] raw_tile_6_3_val;
input  [22:0] raw_tile_6_4_val;
input  [22:0] raw_tile_6_5_val;
input  [22:0] raw_tile_6_6_val;
input  [22:0] raw_tile_6_7_val;
input  [22:0] raw_tile_7_0_val;
input  [22:0] raw_tile_7_1_val;
input  [22:0] raw_tile_7_2_val;
input  [22:0] raw_tile_7_3_val;
input  [22:0] raw_tile_7_4_val;
input  [22:0] raw_tile_7_5_val;
input  [22:0] raw_tile_7_6_val;
input  [22:0] raw_tile_7_7_val;
input  [0:0] write_cache_stream;
output  [9:0] vq_buf_0_address1;
output   vq_buf_0_ce1;
output  [7:0] vq_buf_0_we1;
output  [63:0] vq_buf_0_d1;
output  [9:0] vq_buf_1_address1;
output   vq_buf_1_ce1;
output  [7:0] vq_buf_1_we1;
output  [63:0] vq_buf_1_d1;
output  [9:0] vq_buf_2_address1;
output   vq_buf_2_ce1;
output  [7:0] vq_buf_2_we1;
output  [63:0] vq_buf_2_d1;
output  [9:0] vq_buf_3_address1;
output   vq_buf_3_ce1;
output  [7:0] vq_buf_3_we1;
output  [63:0] vq_buf_3_d1;
output  [9:0] vq_buf_4_address1;
output   vq_buf_4_ce1;
output  [7:0] vq_buf_4_we1;
output  [63:0] vq_buf_4_d1;
output  [9:0] vq_buf_5_address1;
output   vq_buf_5_ce1;
output  [7:0] vq_buf_5_we1;
output  [63:0] vq_buf_5_d1;
output  [9:0] vq_buf_6_address1;
output   vq_buf_6_ce1;
output  [7:0] vq_buf_6_we1;
output  [63:0] vq_buf_6_d1;
output  [9:0] vq_buf_7_address1;
output   vq_buf_7_ce1;
output  [7:0] vq_buf_7_we1;
output  [63:0] vq_buf_7_d1;
output  [9:0] vs_buf_address0;
output   vs_buf_ce0;
input  [31:0] vs_buf_q0;
output  [9:0] vs_buf_address1;
output   vs_buf_ce1;
output   vs_buf_we1;
output  [31:0] vs_buf_d1;
output  [63:0] vq_cache_o_stream_TDATA;
output   vq_cache_o_stream_TVALID;
output  [7:0] vs_cache_o_stream_TDATA;
output   vs_cache_o_stream_TVALID;

reg ap_idle;
reg vq_buf_0_ce1;
reg[7:0] vq_buf_0_we1;
reg vq_buf_1_ce1;
reg[7:0] vq_buf_1_we1;
reg vq_buf_2_ce1;
reg[7:0] vq_buf_2_we1;
reg vq_buf_3_ce1;
reg[7:0] vq_buf_3_we1;
reg vq_buf_4_ce1;
reg[7:0] vq_buf_4_we1;
reg vq_buf_5_ce1;
reg[7:0] vq_buf_5_we1;
reg vq_buf_6_ce1;
reg[7:0] vq_buf_6_we1;
reg vq_buf_7_ce1;
reg[7:0] vq_buf_7_we1;
reg vs_buf_ce0;
reg vs_buf_ce1;
reg vs_buf_we1;
reg vq_cache_o_stream_TVALID;
reg vs_cache_o_stream_TVALID;

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
reg   [2:0] ap_CS_iter7_fsm;
wire    ap_CS_iter7_fsm_state0;
reg   [1:0] ap_CS_iter8_fsm;
wire    ap_CS_iter8_fsm_state0;
wire    ap_CS_iter0_fsm_state2;
wire    ap_CS_iter1_fsm_state3;
wire    ap_CS_iter1_fsm_state4;
wire    ap_CS_iter2_fsm_state5;
wire    ap_CS_iter2_fsm_state6;
wire    ap_CS_iter3_fsm_state7;
wire    ap_CS_iter3_fsm_state8;
wire    ap_CS_iter4_fsm_state9;
wire    ap_CS_iter4_fsm_state10;
wire    ap_CS_iter5_fsm_state11;
wire    ap_CS_iter5_fsm_state12;
wire    ap_CS_iter6_fsm_state13;
wire    ap_CS_iter6_fsm_state14;
wire    ap_CS_iter7_fsm_state15;
reg   [0:0] icmp_ln144_reg_3238;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter6_reg;
reg    ap_predicate_op408_write_state16;
reg    ap_predicate_op410_write_state16;
reg    ap_block_state16_pp0_stage1_iter7;
reg    ap_block_state16_io;
wire    ap_CS_iter7_fsm_state16;
wire    ap_CS_iter8_fsm_state17;
reg    ap_condition_exit_pp0_iter0_stage1;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    vq_cache_o_stream_TDATA_blk_n;
reg    vs_cache_o_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [5:0] mul_fu_1120_p3;
reg   [5:0] mul_reg_3228;
reg   [3:0] cp_1_reg_3233;
reg   [3:0] cp_1_reg_3233_pp0_iter0_reg;
reg   [3:0] cp_1_reg_3233_pp0_iter1_reg;
reg   [3:0] cp_1_reg_3233_pp0_iter2_reg;
reg   [3:0] cp_1_reg_3233_pp0_iter3_reg;
wire   [0:0] icmp_ln144_fu_1146_p2;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter0_reg;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter1_reg;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter2_reg;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter3_reg;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter4_reg;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter5_reg;
reg   [0:0] icmp_ln144_reg_3238_pp0_iter7_reg;
wire   [2:0] trunc_ln144_fu_1158_p1;
reg   [2:0] trunc_ln144_reg_3242;
reg   [2:0] trunc_ln144_reg_3242_pp0_iter0_reg;
reg   [2:0] trunc_ln144_reg_3242_pp0_iter1_reg;
reg   [2:0] trunc_ln144_reg_3242_pp0_iter2_reg;
wire   [22:0] q_val_fu_1162_p19;
reg   [22:0] q_val_reg_3253;
reg   [22:0] q_val_reg_3253_pp0_iter0_reg;
reg   [22:0] q_val_reg_3253_pp0_iter1_reg;
reg   [22:0] q_val_reg_3253_pp0_iter2_reg;
reg   [22:0] q_val_reg_3253_pp0_iter3_reg;
reg   [22:0] q_val_reg_3253_pp0_iter4_reg;
reg   [22:0] q_val_reg_3253_pp0_iter5_reg;
wire   [22:0] sub_ln150_fu_1202_p2;
reg   [22:0] sub_ln150_reg_3261;
reg   [0:0] tmp_reg_3266;
wire   [21:0] abs_max_fu_1236_p3;
reg   [21:0] abs_max_reg_3271;
wire   [22:0] q_val_4_fu_1244_p19;
reg   [22:0] q_val_4_reg_3276;
reg   [22:0] q_val_4_reg_3276_pp0_iter1_reg;
reg   [22:0] q_val_4_reg_3276_pp0_iter2_reg;
reg   [22:0] q_val_4_reg_3276_pp0_iter3_reg;
reg   [22:0] q_val_4_reg_3276_pp0_iter4_reg;
reg   [22:0] q_val_4_reg_3276_pp0_iter5_reg;
wire   [22:0] sub_ln150_1_fu_1275_p2;
reg   [22:0] sub_ln150_1_reg_3284;
reg   [0:0] tmp_2_reg_3289;
wire   [22:0] abs_max_1_fu_1303_p3;
reg   [22:0] abs_max_1_reg_3294;
wire   [22:0] q_val_8_fu_1311_p19;
reg   [22:0] q_val_8_reg_3300;
reg   [22:0] q_val_8_reg_3300_pp0_iter1_reg;
reg   [22:0] q_val_8_reg_3300_pp0_iter2_reg;
reg   [22:0] q_val_8_reg_3300_pp0_iter3_reg;
reg   [22:0] q_val_8_reg_3300_pp0_iter4_reg;
reg   [22:0] q_val_8_reg_3300_pp0_iter5_reg;
wire   [22:0] sub_ln150_2_fu_1342_p2;
reg   [22:0] sub_ln150_2_reg_3308;
reg   [0:0] tmp_3_reg_3313;
wire   [22:0] abs_max_2_fu_1366_p3;
reg   [22:0] abs_max_2_reg_3318;
wire   [22:0] q_val_16_fu_1373_p19;
reg   [22:0] q_val_16_reg_3324;
reg   [22:0] q_val_16_reg_3324_pp0_iter2_reg;
reg   [22:0] q_val_16_reg_3324_pp0_iter3_reg;
reg   [22:0] q_val_16_reg_3324_pp0_iter4_reg;
reg   [22:0] q_val_16_reg_3324_pp0_iter5_reg;
wire   [22:0] sub_ln150_3_fu_1404_p2;
reg   [22:0] sub_ln150_3_reg_3332;
reg   [0:0] tmp_4_reg_3337;
wire   [22:0] abs_max_3_fu_1428_p3;
reg   [22:0] abs_max_3_reg_3342;
wire   [22:0] q_val_1_fu_1435_p19;
reg   [22:0] q_val_1_reg_3348;
reg   [22:0] q_val_1_reg_3348_pp0_iter2_reg;
reg   [22:0] q_val_1_reg_3348_pp0_iter3_reg;
reg   [22:0] q_val_1_reg_3348_pp0_iter4_reg;
reg   [22:0] q_val_1_reg_3348_pp0_iter5_reg;
wire   [22:0] sub_ln150_4_fu_1466_p2;
reg   [22:0] sub_ln150_4_reg_3356;
reg   [0:0] tmp_5_reg_3361;
wire   [22:0] abs_max_4_fu_1490_p3;
reg   [22:0] abs_max_4_reg_3366;
wire   [22:0] q_val_5_fu_1497_p19;
reg   [22:0] q_val_5_reg_3372;
reg   [22:0] q_val_5_reg_3372_pp0_iter3_reg;
reg   [22:0] q_val_5_reg_3372_pp0_iter4_reg;
reg   [22:0] q_val_5_reg_3372_pp0_iter5_reg;
wire   [22:0] sub_ln150_5_fu_1528_p2;
reg   [22:0] sub_ln150_5_reg_3380;
reg   [0:0] tmp_6_reg_3385;
wire   [22:0] abs_max_5_fu_1552_p3;
reg   [22:0] abs_max_5_reg_3390;
wire   [22:0] q_val_9_fu_1559_p19;
reg   [22:0] q_val_9_reg_3396;
reg   [22:0] q_val_9_reg_3396_pp0_iter3_reg;
reg   [22:0] q_val_9_reg_3396_pp0_iter4_reg;
reg   [22:0] q_val_9_reg_3396_pp0_iter5_reg;
wire   [22:0] sub_ln150_6_fu_1590_p2;
reg   [22:0] sub_ln150_6_reg_3404;
reg   [0:0] tmp_7_reg_3409;
wire   [22:0] q_val_2_fu_1604_p19;
reg   [22:0] q_val_2_reg_3414;
reg   [22:0] q_val_2_reg_3414_pp0_iter3_reg;
reg   [22:0] q_val_2_reg_3414_pp0_iter4_reg;
reg   [22:0] q_val_2_reg_3414_pp0_iter5_reg;
reg   [0:0] tmp_8_reg_3423;
wire   [22:0] abs_max_6_fu_1653_p3;
reg   [22:0] abs_max_6_reg_3428;
wire   [22:0] select_ln150_14_fu_1665_p3;
reg   [22:0] select_ln150_14_reg_3434;
wire   [5:0] add_ln146_fu_1674_p2;
reg   [5:0] add_ln146_reg_3440;
reg   [5:0] add_ln146_reg_3440_pp0_iter4_reg;
reg   [5:0] add_ln146_reg_3440_pp0_iter5_reg;
reg   [2:0] lshr_ln_reg_3446;
wire   [22:0] abs_max_8_fu_1693_p3;
reg   [22:0] abs_max_8_reg_3451;
wire   [63:0] zext_ln169_fu_1711_p1;
reg   [63:0] zext_ln169_reg_3458;
reg   [63:0] zext_ln169_reg_3458_pp0_iter5_reg;
reg   [63:0] zext_ln169_reg_3458_pp0_iter6_reg;
reg   [63:0] zext_ln169_reg_3458_pp0_iter7_reg;
reg   [9:0] vs_buf_addr_reg_3472;
reg   [9:0] vs_buf_addr_reg_3472_pp0_iter5_reg;
wire   [0:0] trunc_ln10_fu_1733_p1;
reg   [0:0] trunc_ln10_reg_3478;
reg   [0:0] tmp_9_reg_3483;
wire   [0:0] tmp_9_reg_3483_pp0_iter4_reg;
reg   [0:0] tmp_10_reg_3487;
wire   [0:0] tmp_10_reg_3487_pp0_iter4_reg;
reg   [0:0] tmp_11_reg_3491;
wire   [0:0] tmp_11_reg_3491_pp0_iter4_reg;
reg   [0:0] tmp_12_reg_3495;
wire   [0:0] tmp_12_reg_3495_pp0_iter4_reg;
reg   [0:0] tmp_13_reg_3499;
wire   [0:0] tmp_13_reg_3499_pp0_iter4_reg;
reg   [0:0] tmp_14_reg_3503;
wire   [0:0] tmp_14_reg_3503_pp0_iter4_reg;
reg   [0:0] tmp_15_reg_3507;
wire   [0:0] tmp_15_reg_3507_pp0_iter4_reg;
reg   [0:0] tmp_16_reg_3511;
wire   [0:0] tmp_16_reg_3511_pp0_iter4_reg;
reg   [0:0] tmp_17_reg_3515;
wire   [0:0] tmp_17_reg_3515_pp0_iter4_reg;
reg   [0:0] tmp_18_reg_3519;
wire   [0:0] tmp_18_reg_3519_pp0_iter4_reg;
reg   [0:0] tmp_19_reg_3523;
wire   [0:0] tmp_19_reg_3523_pp0_iter4_reg;
reg   [0:0] tmp_20_reg_3527;
wire   [0:0] tmp_20_reg_3527_pp0_iter4_reg;
reg   [0:0] tmp_21_reg_3531;
wire   [0:0] tmp_21_reg_3531_pp0_iter4_reg;
reg   [0:0] tmp_22_reg_3535;
wire   [0:0] tmp_22_reg_3535_pp0_iter4_reg;
reg   [0:0] tmp_23_reg_3539;
wire   [0:0] tmp_23_reg_3539_pp0_iter4_reg;
reg   [0:0] tmp_24_reg_3543;
wire   [0:0] tmp_24_reg_3543_pp0_iter4_reg;
reg   [0:0] tmp_25_reg_3547;
wire   [0:0] tmp_25_reg_3547_pp0_iter4_reg;
reg   [0:0] tmp_26_reg_3551;
wire   [0:0] tmp_26_reg_3551_pp0_iter4_reg;
reg   [0:0] tmp_27_reg_3555;
wire   [0:0] tmp_27_reg_3555_pp0_iter4_reg;
reg   [0:0] tmp_28_reg_3559;
wire   [0:0] tmp_28_reg_3559_pp0_iter4_reg;
reg   [0:0] tmp_29_reg_3563;
wire   [0:0] tmp_29_reg_3563_pp0_iter4_reg;
reg   [0:0] tmp_30_reg_3567;
wire   [0:0] tmp_30_reg_3567_pp0_iter4_reg;
wire   [5:0] select_ln16_fu_1913_p3;
wire   [2:0] trunc_ln169_fu_1920_p1;
reg   [2:0] trunc_ln169_reg_3576;
reg   [2:0] trunc_ln169_reg_3576_pp0_iter5_reg;
reg   [2:0] trunc_ln169_reg_3576_pp0_iter6_reg;
wire   [31:0] zext_ln172_fu_1931_p1;
reg   [31:0] zext_ln172_reg_3581;
wire   [31:0] shl_ln172_fu_1935_p2;
reg   [31:0] shl_ln172_reg_3586;
reg   [31:0] shl_ln172_reg_3586_pp0_iter5_reg;
wire   [3:0] scale_fu_1989_p3;
reg   [3:0] scale_reg_3591;
reg   [3:0] scale_reg_3591_pp0_iter6_reg;
wire  signed [4:0] sub_i_i_fu_2001_p2;
reg  signed [4:0] sub_i_i_reg_3597;
reg   [0:0] tmp_33_reg_3602;
wire   [4:0] conv_i_i9_i_fu_2015_p2;
reg   [4:0] conv_i_i9_i_reg_3614;
reg   [31:0] vs_buf_load_reg_3619;
wire   [0:0] addr_cmp_fu_2024_p2;
reg   [0:0] addr_cmp_reg_3624;
wire   [31:0] shl_ln172_1_fu_2033_p2;
reg   [31:0] shl_ln172_1_reg_3629;
wire   [0:0] lnot_i_i_fu_2042_p2;
reg   [0:0] lnot_i_i_reg_3634;
wire   [5:0] shl_ln169_fu_2053_p2;
reg   [5:0] shl_ln169_reg_3638;
reg   [5:0] shl_ln169_reg_3638_pp0_iter6_reg;
wire   [22:0] shl_ln163_fu_2062_p2;
reg   [22:0] shl_ln163_reg_3643;
wire   [22:0] ashr_ln163_fu_2071_p2;
reg   [22:0] ashr_ln163_reg_3648;
wire   [22:0] shl_ln163_1_fu_2080_p2;
reg   [22:0] shl_ln163_1_reg_3653;
wire   [22:0] ashr_ln163_1_fu_2089_p2;
reg   [22:0] ashr_ln163_1_reg_3658;
wire   [22:0] shl_ln163_2_fu_2098_p2;
reg   [22:0] shl_ln163_2_reg_3663;
wire   [22:0] ashr_ln163_2_fu_2107_p2;
reg   [22:0] ashr_ln163_2_reg_3668;
wire   [22:0] shl_ln163_3_fu_2116_p2;
reg   [22:0] shl_ln163_3_reg_3673;
wire   [22:0] ashr_ln163_3_fu_2125_p2;
reg   [22:0] ashr_ln163_3_reg_3678;
wire   [22:0] shl_ln163_4_fu_2134_p2;
reg   [22:0] shl_ln163_4_reg_3683;
wire   [22:0] ashr_ln163_4_fu_2143_p2;
reg   [22:0] ashr_ln163_4_reg_3688;
wire   [22:0] shl_ln163_5_fu_2152_p2;
reg   [22:0] shl_ln163_5_reg_3693;
wire   [22:0] ashr_ln163_5_fu_2161_p2;
reg   [22:0] ashr_ln163_5_reg_3698;
wire   [22:0] shl_ln163_6_fu_2170_p2;
reg   [22:0] shl_ln163_6_reg_3703;
wire   [22:0] ashr_ln163_6_fu_2179_p2;
reg   [22:0] ashr_ln163_6_reg_3708;
wire   [22:0] shl_ln163_7_fu_2188_p2;
reg   [22:0] shl_ln163_7_reg_3713;
wire   [22:0] ashr_ln163_7_fu_2197_p2;
reg   [22:0] ashr_ln163_7_reg_3718;
wire  signed [22:0] sext_ln165_fu_2254_p1;
wire  signed [22:0] sext_ln165_1_fu_2279_p1;
wire  signed [22:0] sext_ln165_2_fu_2304_p1;
wire  signed [22:0] sext_ln165_3_fu_2329_p1;
wire  signed [22:0] sext_ln165_4_fu_2354_p1;
wire  signed [22:0] sext_ln165_5_fu_2379_p1;
wire  signed [22:0] sext_ln165_6_fu_2404_p1;
wire  signed [22:0] sext_ln165_7_fu_2429_p1;
wire   [7:0] q_out_8_fu_2473_p3;
reg   [7:0] q_out_8_reg_3763;
wire   [7:0] q_out_9_fu_2521_p3;
reg   [7:0] q_out_9_reg_3769;
wire   [7:0] q_out_10_fu_2569_p3;
reg   [7:0] q_out_10_reg_3775;
wire   [7:0] q_out_11_fu_2617_p3;
reg   [7:0] q_out_11_reg_3781;
wire   [7:0] q_out_12_fu_2665_p3;
reg   [7:0] q_out_12_reg_3787;
wire   [7:0] q_out_13_fu_2713_p3;
reg   [7:0] q_out_13_reg_3793;
wire   [7:0] q_out_14_fu_2761_p3;
reg   [7:0] q_out_14_reg_3799;
wire   [7:0] q_out_fu_2809_p3;
reg   [7:0] q_out_reg_3805;
wire   [63:0] shl_ln169_1_fu_2823_p2;
reg   [63:0] shl_ln169_1_reg_3811;
wire   [7:0] shl_ln169_2_fu_2832_p2;
reg   [7:0] shl_ln169_2_reg_3816;
wire   [63:0] shl_ln169_3_fu_2841_p2;
reg   [63:0] shl_ln169_3_reg_3828;
wire   [63:0] shl_ln169_4_fu_2850_p2;
reg   [63:0] shl_ln169_4_reg_3833;
wire   [63:0] shl_ln169_5_fu_2859_p2;
reg   [63:0] shl_ln169_5_reg_3838;
wire   [63:0] shl_ln169_6_fu_2868_p2;
reg   [63:0] shl_ln169_6_reg_3843;
wire   [63:0] shl_ln169_7_fu_2877_p2;
reg   [63:0] shl_ln169_7_reg_3848;
wire   [63:0] shl_ln169_8_fu_2886_p2;
reg   [63:0] shl_ln169_8_reg_3853;
wire   [63:0] shl_ln169_9_fu_2895_p2;
reg   [63:0] shl_ln169_9_reg_3858;
wire   [5:0] ap_phi_reg_pp0_iter0_s_val_reg_953;
reg   [5:0] ap_phi_reg_pp0_iter1_s_val_reg_953;
reg   [5:0] ap_phi_reg_pp0_iter2_s_val_reg_953;
reg   [5:0] ap_phi_reg_pp0_iter3_s_val_reg_953;
reg   [5:0] ap_phi_reg_pp0_iter4_s_val_reg_953;
reg   [5:0] ap_phi_reg_pp0_iter5_s_val_reg_953;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0_6191_reg_1048;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0_6191_reg_1048;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0_4165167189_reg_1057;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0_4165167189_reg_1057;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0_2147149163169187_reg_1066;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0_2147149163169187_reg_1066;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0137139145151161171185_reg_1075;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0137139145151161171185_reg_1075;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0_1141143153159173183_reg_1084;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0_1141143153159173183_reg_1084;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0_3155157175181_reg_1093;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0_3155157175181_reg_1093;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_0_5177179_reg_1102;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_0_5177179_reg_1102;
wire   [22:0] ap_phi_reg_pp0_iter0_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter1_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter2_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter3_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter4_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter5_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter6_q_val_32_reg_1111;
reg   [22:0] ap_phi_reg_pp0_iter7_q_val_32_reg_1111;
reg   [63:0] reuse_addr_reg_fu_380;
wire    ap_loop_init;
reg   [31:0] reuse_reg_fu_384;
wire   [31:0] or_ln172_fu_2222_p2;
reg   [3:0] cp_fu_388;
wire   [3:0] add_ln144_fu_1152_p2;
reg   [3:0] ap_sig_allocacmp_cp_1;
wire   [22:0] q_val_fu_1162_p17;
wire   [2:0] q_val_fu_1162_p18;
wire   [22:0] select_ln150_fu_1221_p3;
wire   [0:0] icmp_ln224_fu_1230_p2;
wire   [21:0] trunc_ln224_fu_1226_p1;
wire   [22:0] q_val_4_fu_1244_p17;
wire   [22:0] zext_ln147_fu_1289_p1;
wire   [22:0] select_ln150_2_fu_1292_p3;
wire   [0:0] icmp_ln224_1_fu_1297_p2;
wire   [22:0] q_val_8_fu_1311_p17;
wire   [22:0] select_ln150_4_fu_1356_p3;
wire   [0:0] icmp_ln224_2_fu_1361_p2;
wire   [22:0] q_val_16_fu_1373_p17;
wire   [22:0] select_ln150_6_fu_1418_p3;
wire   [0:0] icmp_ln224_3_fu_1423_p2;
wire   [22:0] q_val_1_fu_1435_p17;
wire   [22:0] select_ln150_8_fu_1480_p3;
wire   [0:0] icmp_ln224_4_fu_1485_p2;
wire   [22:0] q_val_5_fu_1497_p17;
wire   [22:0] select_ln150_10_fu_1542_p3;
wire   [0:0] icmp_ln224_5_fu_1547_p2;
wire   [22:0] q_val_9_fu_1559_p17;
wire   [22:0] q_val_2_fu_1604_p17;
wire   [22:0] select_ln150_12_fu_1643_p3;
wire   [0:0] icmp_ln224_6_fu_1648_p2;
wire   [22:0] sub_ln150_7_fu_1660_p2;
wire   [5:0] zext_ln144_fu_1671_p1;
wire   [0:0] icmp_ln224_7_fu_1689_p2;
wire   [9:0] tmp_s_fu_1699_p3;
wire   [9:0] add_ln169_fu_1706_p2;
wire   [0:0] icmp_ln12_fu_1716_p2;
wire   [22:0] x_1_fu_1721_p2;
wire   [22:0] x_2_fu_1726_p3;
wire   [4:0] shl_ln1_fu_1923_p3;
wire   [1:0] tmp_32_fu_1949_p4;
wire   [0:0] tmp_31_fu_1941_p3;
wire   [0:0] xor_ln154_fu_1969_p2;
wire   [0:0] icmp_ln43_fu_1959_p2;
wire   [0:0] or_ln154_fu_1983_p2;
wire   [3:0] select_ln154_fu_1975_p3;
wire   [3:0] trunc_ln154_fu_1965_p1;
wire   [4:0] scale_cast_fu_1997_p1;
wire   [31:0] zext_ln172_1_fu_2029_p1;
wire  signed [31:0] conv_i_i9_i_cast_fu_2047_p1;
wire   [22:0] conv_i_i9_i_castcast_fu_2058_p1;
wire  signed [31:0] conv_i_i_i41_fu_2050_p1;
wire   [22:0] conv_i_i_i41cast_fu_2067_p1;
wire   [22:0] conv_i_i9_i_castcast384_fu_2076_p1;
wire   [22:0] conv_i_i_i41cast385_fu_2085_p1;
wire   [22:0] conv_i_i9_i_castcast386_fu_2094_p1;
wire   [22:0] conv_i_i_i41cast387_fu_2103_p1;
wire   [22:0] conv_i_i9_i_castcast388_fu_2112_p1;
wire   [22:0] conv_i_i_i41cast389_fu_2121_p1;
wire   [22:0] conv_i_i9_i_castcast390_fu_2130_p1;
wire   [22:0] conv_i_i_i41cast391_fu_2139_p1;
wire   [22:0] conv_i_i9_i_castcast392_fu_2148_p1;
wire   [22:0] conv_i_i_i41cast393_fu_2157_p1;
wire   [22:0] conv_i_i9_i_castcast394_fu_2166_p1;
wire   [22:0] conv_i_i_i41cast395_fu_2175_p1;
wire   [22:0] conv_i_i9_i_castcast396_fu_2184_p1;
wire   [22:0] conv_i_i_i41cast397_fu_2193_p1;
wire   [31:0] reuse_select_fu_2205_p3;
wire   [31:0] xor_ln172_fu_2211_p2;
wire   [31:0] and_ln172_fu_2216_p2;
wire   [22:0] q_val_3_fu_2233_p3;
wire   [22:0] q_val_6_fu_2238_p2;
wire   [21:0] q_val_7_fu_2244_p4;
wire   [22:0] q_val_10_fu_2258_p3;
wire   [22:0] q_val_11_fu_2263_p2;
wire   [21:0] q_val_12_fu_2269_p4;
wire   [22:0] q_val_13_fu_2283_p3;
wire   [22:0] q_val_14_fu_2288_p2;
wire   [21:0] q_val_15_fu_2294_p4;
wire   [22:0] q_val_17_fu_2308_p3;
wire   [22:0] q_val_18_fu_2313_p2;
wire   [21:0] q_val_19_fu_2319_p4;
wire   [22:0] q_val_20_fu_2333_p3;
wire   [22:0] q_val_21_fu_2338_p2;
wire   [21:0] q_val_22_fu_2344_p4;
wire   [22:0] q_val_23_fu_2358_p3;
wire   [22:0] q_val_24_fu_2363_p2;
wire   [21:0] q_val_25_fu_2369_p4;
wire   [22:0] q_val_26_fu_2383_p3;
wire   [22:0] q_val_27_fu_2388_p2;
wire   [21:0] q_val_28_fu_2394_p4;
wire   [22:0] q_val_29_fu_2408_p3;
wire   [22:0] q_val_30_fu_2413_p2;
wire   [21:0] q_val_31_fu_2419_p4;
wire   [15:0] tmp_34_fu_2439_p4;
wire   [0:0] icmp_ln42_fu_2433_p2;
wire   [0:0] icmp_ln43_1_fu_2449_p2;
wire   [0:0] or_ln168_fu_2467_p2;
wire   [7:0] select_ln168_fu_2459_p3;
wire   [7:0] trunc_ln168_fu_2455_p1;
wire   [15:0] tmp_35_fu_2487_p4;
wire   [0:0] icmp_ln42_1_fu_2481_p2;
wire   [0:0] icmp_ln43_2_fu_2497_p2;
wire   [0:0] or_ln168_1_fu_2515_p2;
wire   [7:0] select_ln168_2_fu_2507_p3;
wire   [7:0] trunc_ln168_1_fu_2503_p1;
wire   [15:0] tmp_36_fu_2535_p4;
wire   [0:0] icmp_ln42_2_fu_2529_p2;
wire   [0:0] icmp_ln43_3_fu_2545_p2;
wire   [0:0] or_ln168_2_fu_2563_p2;
wire   [7:0] select_ln168_4_fu_2555_p3;
wire   [7:0] trunc_ln168_2_fu_2551_p1;
wire   [15:0] tmp_37_fu_2583_p4;
wire   [0:0] icmp_ln42_3_fu_2577_p2;
wire   [0:0] icmp_ln43_4_fu_2593_p2;
wire   [0:0] or_ln168_3_fu_2611_p2;
wire   [7:0] select_ln168_6_fu_2603_p3;
wire   [7:0] trunc_ln168_3_fu_2599_p1;
wire   [15:0] tmp_38_fu_2631_p4;
wire   [0:0] icmp_ln42_4_fu_2625_p2;
wire   [0:0] icmp_ln43_5_fu_2641_p2;
wire   [0:0] or_ln168_4_fu_2659_p2;
wire   [7:0] select_ln168_8_fu_2651_p3;
wire   [7:0] trunc_ln168_4_fu_2647_p1;
wire   [15:0] tmp_39_fu_2679_p4;
wire   [0:0] icmp_ln42_5_fu_2673_p2;
wire   [0:0] icmp_ln43_6_fu_2689_p2;
wire   [0:0] or_ln168_5_fu_2707_p2;
wire   [7:0] select_ln168_10_fu_2699_p3;
wire   [7:0] trunc_ln168_5_fu_2695_p1;
wire   [15:0] tmp_40_fu_2727_p4;
wire   [0:0] icmp_ln42_6_fu_2721_p2;
wire   [0:0] icmp_ln43_7_fu_2737_p2;
wire   [0:0] or_ln168_6_fu_2755_p2;
wire   [7:0] select_ln168_12_fu_2747_p3;
wire   [7:0] trunc_ln168_6_fu_2743_p1;
wire   [15:0] tmp_41_fu_2775_p4;
wire   [0:0] icmp_ln42_7_fu_2769_p2;
wire   [0:0] icmp_ln43_8_fu_2785_p2;
wire   [0:0] or_ln168_7_fu_2803_p2;
wire   [7:0] select_ln168_14_fu_2795_p3;
wire   [7:0] trunc_ln168_7_fu_2791_p1;
wire   [63:0] zext_ln169_2_fu_2820_p1;
wire   [63:0] zext_ln169_1_fu_2817_p1;
wire   [7:0] zext_ln169_3_fu_2829_p1;
wire   [63:0] zext_ln169_4_fu_2838_p1;
wire   [63:0] zext_ln169_5_fu_2847_p1;
wire   [63:0] zext_ln169_6_fu_2856_p1;
wire   [63:0] zext_ln169_7_fu_2865_p1;
wire   [63:0] zext_ln169_8_fu_2874_p1;
wire   [63:0] zext_ln169_9_fu_2883_p1;
wire   [63:0] zext_ln169_10_fu_2892_p1;
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
reg   [1:0] ap_NS_iter0_fsm;
reg   [2:0] ap_NS_iter1_fsm;
reg   [2:0] ap_NS_iter2_fsm;
reg   [2:0] ap_NS_iter3_fsm;
reg   [2:0] ap_NS_iter4_fsm;
reg   [2:0] ap_NS_iter5_fsm;
reg   [2:0] ap_NS_iter6_fsm;
reg   [2:0] ap_NS_iter7_fsm;
reg   [1:0] ap_NS_iter8_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter1_fsm_state3_blk;
wire    ap_ST_iter1_fsm_state4_blk;
wire    ap_ST_iter2_fsm_state5_blk;
wire    ap_ST_iter2_fsm_state6_blk;
wire    ap_ST_iter3_fsm_state7_blk;
wire    ap_ST_iter3_fsm_state8_blk;
wire    ap_ST_iter4_fsm_state9_blk;
wire    ap_ST_iter4_fsm_state10_blk;
wire    ap_ST_iter5_fsm_state11_blk;
wire    ap_ST_iter5_fsm_state12_blk;
wire    ap_ST_iter6_fsm_state13_blk;
wire    ap_ST_iter6_fsm_state14_blk;
wire    ap_ST_iter7_fsm_state15_blk;
reg    ap_ST_iter7_fsm_state16_blk;
wire    ap_ST_iter8_fsm_state17_blk;
wire    ap_start_int;
reg    ap_condition_2652;
reg    ap_condition_2657;
reg    ap_condition_2663;
reg    ap_condition_2670;
reg    ap_condition_2678;
reg    ap_condition_2687;
reg    ap_condition_2697;
reg    ap_condition_2708;
reg    ap_condition_2720;
reg    ap_condition_2733;
reg    ap_condition_2747;
reg    ap_condition_2762;
reg    ap_condition_2778;
reg    ap_condition_2795;
reg    ap_condition_2813;
reg    ap_condition_2832;
reg    ap_condition_2852;
reg    ap_condition_2873;
reg    ap_condition_2895;
reg    ap_condition_2918;
reg    ap_condition_2942;
reg    ap_condition_2967;
reg    ap_condition_2992;
reg    ap_condition_192;
reg    ap_condition_2998;
reg    ap_condition_394;
reg    ap_condition_385;
wire   [2:0] q_val_fu_1162_p1;
wire   [2:0] q_val_fu_1162_p3;
wire   [2:0] q_val_fu_1162_p5;
wire   [2:0] q_val_fu_1162_p7;
wire  signed [2:0] q_val_fu_1162_p9;
wire  signed [2:0] q_val_fu_1162_p11;
wire  signed [2:0] q_val_fu_1162_p13;
wire  signed [2:0] q_val_fu_1162_p15;
wire   [2:0] q_val_4_fu_1244_p1;
wire   [2:0] q_val_4_fu_1244_p3;
wire   [2:0] q_val_4_fu_1244_p5;
wire   [2:0] q_val_4_fu_1244_p7;
wire  signed [2:0] q_val_4_fu_1244_p9;
wire  signed [2:0] q_val_4_fu_1244_p11;
wire  signed [2:0] q_val_4_fu_1244_p13;
wire  signed [2:0] q_val_4_fu_1244_p15;
wire   [2:0] q_val_8_fu_1311_p1;
wire   [2:0] q_val_8_fu_1311_p3;
wire   [2:0] q_val_8_fu_1311_p5;
wire   [2:0] q_val_8_fu_1311_p7;
wire  signed [2:0] q_val_8_fu_1311_p9;
wire  signed [2:0] q_val_8_fu_1311_p11;
wire  signed [2:0] q_val_8_fu_1311_p13;
wire  signed [2:0] q_val_8_fu_1311_p15;
wire   [2:0] q_val_16_fu_1373_p1;
wire   [2:0] q_val_16_fu_1373_p3;
wire   [2:0] q_val_16_fu_1373_p5;
wire   [2:0] q_val_16_fu_1373_p7;
wire  signed [2:0] q_val_16_fu_1373_p9;
wire  signed [2:0] q_val_16_fu_1373_p11;
wire  signed [2:0] q_val_16_fu_1373_p13;
wire  signed [2:0] q_val_16_fu_1373_p15;
wire   [2:0] q_val_1_fu_1435_p1;
wire   [2:0] q_val_1_fu_1435_p3;
wire   [2:0] q_val_1_fu_1435_p5;
wire   [2:0] q_val_1_fu_1435_p7;
wire  signed [2:0] q_val_1_fu_1435_p9;
wire  signed [2:0] q_val_1_fu_1435_p11;
wire  signed [2:0] q_val_1_fu_1435_p13;
wire  signed [2:0] q_val_1_fu_1435_p15;
wire   [2:0] q_val_5_fu_1497_p1;
wire   [2:0] q_val_5_fu_1497_p3;
wire   [2:0] q_val_5_fu_1497_p5;
wire   [2:0] q_val_5_fu_1497_p7;
wire  signed [2:0] q_val_5_fu_1497_p9;
wire  signed [2:0] q_val_5_fu_1497_p11;
wire  signed [2:0] q_val_5_fu_1497_p13;
wire  signed [2:0] q_val_5_fu_1497_p15;
wire   [2:0] q_val_9_fu_1559_p1;
wire   [2:0] q_val_9_fu_1559_p3;
wire   [2:0] q_val_9_fu_1559_p5;
wire   [2:0] q_val_9_fu_1559_p7;
wire  signed [2:0] q_val_9_fu_1559_p9;
wire  signed [2:0] q_val_9_fu_1559_p11;
wire  signed [2:0] q_val_9_fu_1559_p13;
wire  signed [2:0] q_val_9_fu_1559_p15;
wire   [2:0] q_val_2_fu_1604_p1;
wire   [2:0] q_val_2_fu_1604_p3;
wire   [2:0] q_val_2_fu_1604_p5;
wire   [2:0] q_val_2_fu_1604_p7;
wire  signed [2:0] q_val_2_fu_1604_p9;
wire  signed [2:0] q_val_2_fu_1604_p11;
wire  signed [2:0] q_val_2_fu_1604_p13;
wire  signed [2:0] q_val_2_fu_1604_p15;
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
//#0 ap_CS_iter7_fsm = 3'd1;
//#0 ap_CS_iter8_fsm = 2'd1;
//#0 reuse_addr_reg_fu_380 = 64'd0;
//#0 reuse_reg_fu_384 = 32'd0;
//#0 cp_fu_388 = 4'd0;
//#0 ap_done_reg = 1'b0;
end

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U95(
    .din0(raw_tile_0_0_val),
    .din1(raw_tile_1_0_val),
    .din2(raw_tile_2_0_val),
    .din3(raw_tile_3_0_val),
    .din4(raw_tile_4_0_val),
    .din5(raw_tile_5_0_val),
    .din6(raw_tile_6_0_val),
    .din7(raw_tile_7_0_val),
    .def(q_val_fu_1162_p17),
    .sel(q_val_fu_1162_p18),
    .dout(q_val_fu_1162_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U96(
    .din0(raw_tile_0_1_val),
    .din1(raw_tile_1_1_val),
    .din2(raw_tile_2_1_val),
    .din3(raw_tile_3_1_val),
    .din4(raw_tile_4_1_val),
    .din5(raw_tile_5_1_val),
    .din6(raw_tile_6_1_val),
    .din7(raw_tile_7_1_val),
    .def(q_val_4_fu_1244_p17),
    .sel(trunc_ln144_reg_3242),
    .dout(q_val_4_fu_1244_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U97(
    .din0(raw_tile_0_2_val),
    .din1(raw_tile_1_2_val),
    .din2(raw_tile_2_2_val),
    .din3(raw_tile_3_2_val),
    .din4(raw_tile_4_2_val),
    .din5(raw_tile_5_2_val),
    .din6(raw_tile_6_2_val),
    .din7(raw_tile_7_2_val),
    .def(q_val_8_fu_1311_p17),
    .sel(trunc_ln144_reg_3242_pp0_iter0_reg),
    .dout(q_val_8_fu_1311_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U98(
    .din0(raw_tile_0_3_val),
    .din1(raw_tile_1_3_val),
    .din2(raw_tile_2_3_val),
    .din3(raw_tile_3_3_val),
    .din4(raw_tile_4_3_val),
    .din5(raw_tile_5_3_val),
    .din6(raw_tile_6_3_val),
    .din7(raw_tile_7_3_val),
    .def(q_val_16_fu_1373_p17),
    .sel(trunc_ln144_reg_3242_pp0_iter0_reg),
    .dout(q_val_16_fu_1373_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U99(
    .din0(raw_tile_0_4_val),
    .din1(raw_tile_1_4_val),
    .din2(raw_tile_2_4_val),
    .din3(raw_tile_3_4_val),
    .din4(raw_tile_4_4_val),
    .din5(raw_tile_5_4_val),
    .din6(raw_tile_6_4_val),
    .din7(raw_tile_7_4_val),
    .def(q_val_1_fu_1435_p17),
    .sel(trunc_ln144_reg_3242_pp0_iter1_reg),
    .dout(q_val_1_fu_1435_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U100(
    .din0(raw_tile_0_5_val),
    .din1(raw_tile_1_5_val),
    .din2(raw_tile_2_5_val),
    .din3(raw_tile_3_5_val),
    .din4(raw_tile_4_5_val),
    .din5(raw_tile_5_5_val),
    .din6(raw_tile_6_5_val),
    .din7(raw_tile_7_5_val),
    .def(q_val_5_fu_1497_p17),
    .sel(trunc_ln144_reg_3242_pp0_iter1_reg),
    .dout(q_val_5_fu_1497_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U101(
    .din0(raw_tile_0_6_val),
    .din1(raw_tile_1_6_val),
    .din2(raw_tile_2_6_val),
    .din3(raw_tile_3_6_val),
    .din4(raw_tile_4_6_val),
    .din5(raw_tile_5_6_val),
    .din6(raw_tile_6_6_val),
    .din7(raw_tile_7_6_val),
    .def(q_val_9_fu_1559_p17),
    .sel(trunc_ln144_reg_3242_pp0_iter2_reg),
    .dout(q_val_9_fu_1559_p19)
);

RV_GEMM_sparsemux_17_3_23_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .CASE0( 3'h0 ),
    .din0_WIDTH( 23 ),
    .CASE1( 3'h1 ),
    .din1_WIDTH( 23 ),
    .CASE2( 3'h2 ),
    .din2_WIDTH( 23 ),
    .CASE3( 3'h3 ),
    .din3_WIDTH( 23 ),
    .CASE4( 3'h4 ),
    .din4_WIDTH( 23 ),
    .CASE5( 3'h5 ),
    .din5_WIDTH( 23 ),
    .CASE6( 3'h6 ),
    .din6_WIDTH( 23 ),
    .CASE7( 3'h7 ),
    .din7_WIDTH( 23 ),
    .def_WIDTH( 23 ),
    .sel_WIDTH( 3 ),
    .dout_WIDTH( 23 ))
sparsemux_17_3_23_1_1_U102(
    .din0(raw_tile_0_7_val),
    .din1(raw_tile_1_7_val),
    .din2(raw_tile_2_7_val),
    .din3(raw_tile_3_7_val),
    .din4(raw_tile_4_7_val),
    .din5(raw_tile_5_7_val),
    .din6(raw_tile_6_7_val),
    .din7(raw_tile_7_7_val),
    .def(q_val_2_fu_1604_p17),
    .sel(trunc_ln144_reg_3242_pp0_iter2_reg),
    .dout(q_val_2_fu_1604_p19)
);

RV_GEMM_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_CS_iter8_fsm <= ap_ST_iter8_fsm_state0;
    end else begin
        ap_CS_iter8_fsm <= ap_NS_iter8_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter8_fsm_state17) & (ap_loop_exit_ready_pp0_iter8_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter8_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter7_fsm_state16))) begin
        ap_loop_exit_ready_pp0_iter8_reg <= ap_loop_exit_ready_pp0_iter7_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2992)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= select_ln16_fu_1913_p3;
        end else if ((1'b1 == ap_condition_2967)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd59;
        end else if ((1'b1 == ap_condition_2942)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd60;
        end else if ((1'b1 == ap_condition_2918)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd61;
        end else if ((1'b1 == ap_condition_2895)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd62;
        end else if ((1'b1 == ap_condition_2873)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd63;
        end else if ((1'b1 == ap_condition_2852)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd0;
        end else if ((1'b1 == ap_condition_2832)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd1;
        end else if ((1'b1 == ap_condition_2813)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd2;
        end else if ((1'b1 == ap_condition_2795)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd3;
        end else if ((1'b1 == ap_condition_2778)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd4;
        end else if ((1'b1 == ap_condition_2762)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd5;
        end else if ((1'b1 == ap_condition_2747)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd6;
        end else if ((1'b1 == ap_condition_2733)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd7;
        end else if ((1'b1 == ap_condition_2720)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd8;
        end else if ((1'b1 == ap_condition_2708)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd9;
        end else if ((1'b1 == ap_condition_2697)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd10;
        end else if ((1'b1 == ap_condition_2687)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd11;
        end else if ((1'b1 == ap_condition_2678)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd12;
        end else if ((1'b1 == ap_condition_2670)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd13;
        end else if ((1'b1 == ap_condition_2663)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd14;
        end else if ((1'b1 == ap_condition_2657)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd15;
        end else if ((1'b1 == ap_condition_2652)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= 6'd16;
        end else if ((1'b1 == ap_CS_iter4_fsm_state10)) begin
            ap_phi_reg_pp0_iter5_s_val_reg_953 <= ap_phi_reg_pp0_iter4_s_val_reg_953;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0137139145151161171185_reg_1075 <= q_val_reg_3253_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter5_q_val_0137139145151161171185_reg_1075;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0_1141143153159173183_reg_1084 <= q_val_4_reg_3276_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter5_q_val_0_1141143153159173183_reg_1084;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0_2147149163169187_reg_1066 <= q_val_8_reg_3300_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter5_q_val_0_2147149163169187_reg_1066;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0_3155157175181_reg_1093 <= q_val_16_reg_3324_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter5_q_val_0_3155157175181_reg_1093;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0_4165167189_reg_1057 <= q_val_1_reg_3348_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter5_q_val_0_4165167189_reg_1057;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0_5177179_reg_1102 <= q_val_5_reg_3372_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter5_q_val_0_5177179_reg_1102;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_0_6191_reg_1048 <= q_val_9_reg_3396_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter5_q_val_0_6191_reg_1048;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_192)) begin
        if ((1'b1 == ap_condition_2998)) begin
            ap_phi_reg_pp0_iter6_q_val_32_reg_1111 <= q_val_2_reg_3414_pp0_iter5_reg;
        end else if ((1'b1 == ap_CS_iter5_fsm_state12)) begin
            ap_phi_reg_pp0_iter6_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter5_q_val_32_reg_1111;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0137139145151161171185_reg_1075 <= sext_ln165_fu_2254_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter6_q_val_0137139145151161171185_reg_1075;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0_1141143153159173183_reg_1084 <= sext_ln165_1_fu_2279_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter6_q_val_0_1141143153159173183_reg_1084;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0_2147149163169187_reg_1066 <= sext_ln165_2_fu_2304_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter6_q_val_0_2147149163169187_reg_1066;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0_3155157175181_reg_1093 <= sext_ln165_3_fu_2329_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter6_q_val_0_3155157175181_reg_1093;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0_4165167189_reg_1057 <= sext_ln165_4_fu_2354_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter6_q_val_0_4165167189_reg_1057;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0_5177179_reg_1102 <= sext_ln165_5_fu_2379_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter6_q_val_0_5177179_reg_1102;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_0_6191_reg_1048 <= sext_ln165_6_fu_2404_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter6_q_val_0_6191_reg_1048;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_394)) begin
        if (((icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_reg_3634 == 1'd0))) begin
            ap_phi_reg_pp0_iter7_q_val_32_reg_1111 <= sext_ln165_7_fu_2429_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter7_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter6_q_val_32_reg_1111;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_385)) begin
        if ((icmp_ln144_fu_1146_p2 == 1'd0)) begin
            cp_fu_388 <= add_ln144_fu_1152_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            cp_fu_388 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        reuse_addr_reg_fu_380 <= 64'd18446744073709551615;
    end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state12) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0))) begin
        reuse_addr_reg_fu_380 <= zext_ln169_reg_3458;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        reuse_reg_fu_384 <= 32'd0;
    end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0))) begin
        reuse_reg_fu_384 <= or_ln172_fu_2222_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
        abs_max_1_reg_3294 <= abs_max_1_fu_1303_p3;
        q_val_8_reg_3300 <= q_val_8_fu_1311_p19;
        sub_ln150_2_reg_3308 <= sub_ln150_2_fu_1342_p2;
        tmp_3_reg_3313 <= q_val_8_fu_1311_p19[32'd22];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
        abs_max_2_reg_3318 <= abs_max_2_fu_1366_p3;
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        ap_phi_reg_pp0_iter2_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter1_q_val_0137139145151161171185_reg_1075;
        ap_phi_reg_pp0_iter2_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter1_q_val_0_1141143153159173183_reg_1084;
        ap_phi_reg_pp0_iter2_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter1_q_val_0_2147149163169187_reg_1066;
        ap_phi_reg_pp0_iter2_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter1_q_val_0_3155157175181_reg_1093;
        ap_phi_reg_pp0_iter2_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter1_q_val_0_4165167189_reg_1057;
        ap_phi_reg_pp0_iter2_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter1_q_val_0_5177179_reg_1102;
        ap_phi_reg_pp0_iter2_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter1_q_val_0_6191_reg_1048;
        ap_phi_reg_pp0_iter2_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter1_q_val_32_reg_1111;
        ap_phi_reg_pp0_iter2_s_val_reg_953 <= ap_phi_reg_pp0_iter1_s_val_reg_953;
        cp_1_reg_3233_pp0_iter1_reg <= cp_1_reg_3233_pp0_iter0_reg;
        icmp_ln144_reg_3238_pp0_iter1_reg <= icmp_ln144_reg_3238_pp0_iter0_reg;
        q_val_16_reg_3324 <= q_val_16_fu_1373_p19;
        q_val_4_reg_3276_pp0_iter1_reg <= q_val_4_reg_3276;
        q_val_8_reg_3300_pp0_iter1_reg <= q_val_8_reg_3300;
        q_val_reg_3253_pp0_iter1_reg <= q_val_reg_3253_pp0_iter0_reg;
        sub_ln150_3_reg_3332 <= sub_ln150_3_fu_1404_p2;
        tmp_4_reg_3337 <= q_val_16_fu_1373_p19[32'd22];
        trunc_ln144_reg_3242_pp0_iter1_reg <= trunc_ln144_reg_3242_pp0_iter0_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
        abs_max_3_reg_3342 <= abs_max_3_fu_1428_p3;
        q_val_1_reg_3348 <= q_val_1_fu_1435_p19;
        sub_ln150_4_reg_3356 <= sub_ln150_4_fu_1466_p2;
        tmp_5_reg_3361 <= q_val_1_fu_1435_p19[32'd22];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
        abs_max_4_reg_3366 <= abs_max_4_fu_1490_p3;
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        ap_phi_reg_pp0_iter3_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter2_q_val_0137139145151161171185_reg_1075;
        ap_phi_reg_pp0_iter3_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter2_q_val_0_1141143153159173183_reg_1084;
        ap_phi_reg_pp0_iter3_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter2_q_val_0_2147149163169187_reg_1066;
        ap_phi_reg_pp0_iter3_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter2_q_val_0_3155157175181_reg_1093;
        ap_phi_reg_pp0_iter3_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter2_q_val_0_4165167189_reg_1057;
        ap_phi_reg_pp0_iter3_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter2_q_val_0_5177179_reg_1102;
        ap_phi_reg_pp0_iter3_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter2_q_val_0_6191_reg_1048;
        ap_phi_reg_pp0_iter3_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter2_q_val_32_reg_1111;
        ap_phi_reg_pp0_iter3_s_val_reg_953 <= ap_phi_reg_pp0_iter2_s_val_reg_953;
        cp_1_reg_3233_pp0_iter2_reg <= cp_1_reg_3233_pp0_iter1_reg;
        icmp_ln144_reg_3238_pp0_iter2_reg <= icmp_ln144_reg_3238_pp0_iter1_reg;
        q_val_16_reg_3324_pp0_iter2_reg <= q_val_16_reg_3324;
        q_val_1_reg_3348_pp0_iter2_reg <= q_val_1_reg_3348;
        q_val_4_reg_3276_pp0_iter2_reg <= q_val_4_reg_3276_pp0_iter1_reg;
        q_val_5_reg_3372 <= q_val_5_fu_1497_p19;
        q_val_8_reg_3300_pp0_iter2_reg <= q_val_8_reg_3300_pp0_iter1_reg;
        q_val_reg_3253_pp0_iter2_reg <= q_val_reg_3253_pp0_iter1_reg;
        sub_ln150_5_reg_3380 <= sub_ln150_5_fu_1528_p2;
        tmp_6_reg_3385 <= q_val_5_fu_1497_p19[32'd22];
        trunc_ln144_reg_3242_pp0_iter2_reg <= trunc_ln144_reg_3242_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
        abs_max_5_reg_3390 <= abs_max_5_fu_1552_p3;
        q_val_2_reg_3414 <= q_val_2_fu_1604_p19;
        q_val_9_reg_3396 <= q_val_9_fu_1559_p19;
        sub_ln150_6_reg_3404 <= sub_ln150_6_fu_1590_p2;
        tmp_7_reg_3409 <= q_val_9_fu_1559_p19[32'd22];
        tmp_8_reg_3423 <= q_val_2_fu_1604_p19[32'd22];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
        abs_max_6_reg_3428 <= abs_max_6_fu_1653_p3;
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        ap_phi_reg_pp0_iter4_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter3_q_val_0137139145151161171185_reg_1075;
        ap_phi_reg_pp0_iter4_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter3_q_val_0_1141143153159173183_reg_1084;
        ap_phi_reg_pp0_iter4_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter3_q_val_0_2147149163169187_reg_1066;
        ap_phi_reg_pp0_iter4_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter3_q_val_0_3155157175181_reg_1093;
        ap_phi_reg_pp0_iter4_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter3_q_val_0_4165167189_reg_1057;
        ap_phi_reg_pp0_iter4_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter3_q_val_0_5177179_reg_1102;
        ap_phi_reg_pp0_iter4_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter3_q_val_0_6191_reg_1048;
        ap_phi_reg_pp0_iter4_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter3_q_val_32_reg_1111;
        ap_phi_reg_pp0_iter4_s_val_reg_953 <= ap_phi_reg_pp0_iter3_s_val_reg_953;
        cp_1_reg_3233_pp0_iter3_reg <= cp_1_reg_3233_pp0_iter2_reg;
        icmp_ln144_reg_3238_pp0_iter3_reg <= icmp_ln144_reg_3238_pp0_iter2_reg;
        q_val_16_reg_3324_pp0_iter3_reg <= q_val_16_reg_3324_pp0_iter2_reg;
        q_val_1_reg_3348_pp0_iter3_reg <= q_val_1_reg_3348_pp0_iter2_reg;
        q_val_2_reg_3414_pp0_iter3_reg <= q_val_2_reg_3414;
        q_val_4_reg_3276_pp0_iter3_reg <= q_val_4_reg_3276_pp0_iter2_reg;
        q_val_5_reg_3372_pp0_iter3_reg <= q_val_5_reg_3372;
        q_val_8_reg_3300_pp0_iter3_reg <= q_val_8_reg_3300_pp0_iter2_reg;
        q_val_9_reg_3396_pp0_iter3_reg <= q_val_9_reg_3396;
        q_val_reg_3253_pp0_iter3_reg <= q_val_reg_3253_pp0_iter2_reg;
        select_ln150_14_reg_3434 <= select_ln150_14_fu_1665_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
        abs_max_8_reg_3451 <= abs_max_8_fu_1693_p3;
        add_ln146_reg_3440 <= add_ln146_fu_1674_p2;
        lshr_ln_reg_3446 <= {{add_ln146_fu_1674_p2[5:3]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        abs_max_reg_3271 <= abs_max_fu_1236_p3;
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        ap_phi_reg_pp0_iter1_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter0_q_val_0137139145151161171185_reg_1075;
        ap_phi_reg_pp0_iter1_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter0_q_val_0_1141143153159173183_reg_1084;
        ap_phi_reg_pp0_iter1_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter0_q_val_0_2147149163169187_reg_1066;
        ap_phi_reg_pp0_iter1_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter0_q_val_0_3155157175181_reg_1093;
        ap_phi_reg_pp0_iter1_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter0_q_val_0_4165167189_reg_1057;
        ap_phi_reg_pp0_iter1_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter0_q_val_0_5177179_reg_1102;
        ap_phi_reg_pp0_iter1_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter0_q_val_0_6191_reg_1048;
        ap_phi_reg_pp0_iter1_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter0_q_val_32_reg_1111;
        ap_phi_reg_pp0_iter1_s_val_reg_953 <= ap_phi_reg_pp0_iter0_s_val_reg_953;
        cp_1_reg_3233_pp0_iter0_reg <= cp_1_reg_3233;
        icmp_ln144_reg_3238_pp0_iter0_reg <= icmp_ln144_reg_3238;
        q_val_4_reg_3276 <= q_val_4_fu_1244_p19;
        q_val_reg_3253_pp0_iter0_reg <= q_val_reg_3253;
        sub_ln150_1_reg_3284 <= sub_ln150_1_fu_1275_p2;
        tmp_2_reg_3289 <= q_val_4_fu_1244_p19[32'd22];
        trunc_ln144_reg_3242_pp0_iter0_reg <= trunc_ln144_reg_3242;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        add_ln146_reg_3440_pp0_iter4_reg <= add_ln146_reg_3440;
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        ap_phi_reg_pp0_iter5_q_val_0137139145151161171185_reg_1075 <= ap_phi_reg_pp0_iter4_q_val_0137139145151161171185_reg_1075;
        ap_phi_reg_pp0_iter5_q_val_0_1141143153159173183_reg_1084 <= ap_phi_reg_pp0_iter4_q_val_0_1141143153159173183_reg_1084;
        ap_phi_reg_pp0_iter5_q_val_0_2147149163169187_reg_1066 <= ap_phi_reg_pp0_iter4_q_val_0_2147149163169187_reg_1066;
        ap_phi_reg_pp0_iter5_q_val_0_3155157175181_reg_1093 <= ap_phi_reg_pp0_iter4_q_val_0_3155157175181_reg_1093;
        ap_phi_reg_pp0_iter5_q_val_0_4165167189_reg_1057 <= ap_phi_reg_pp0_iter4_q_val_0_4165167189_reg_1057;
        ap_phi_reg_pp0_iter5_q_val_0_5177179_reg_1102 <= ap_phi_reg_pp0_iter4_q_val_0_5177179_reg_1102;
        ap_phi_reg_pp0_iter5_q_val_0_6191_reg_1048 <= ap_phi_reg_pp0_iter4_q_val_0_6191_reg_1048;
        ap_phi_reg_pp0_iter5_q_val_32_reg_1111 <= ap_phi_reg_pp0_iter4_q_val_32_reg_1111;
        icmp_ln144_reg_3238_pp0_iter4_reg <= icmp_ln144_reg_3238_pp0_iter3_reg;
        q_val_16_reg_3324_pp0_iter4_reg <= q_val_16_reg_3324_pp0_iter3_reg;
        q_val_1_reg_3348_pp0_iter4_reg <= q_val_1_reg_3348_pp0_iter3_reg;
        q_val_2_reg_3414_pp0_iter4_reg <= q_val_2_reg_3414_pp0_iter3_reg;
        q_val_4_reg_3276_pp0_iter4_reg <= q_val_4_reg_3276_pp0_iter3_reg;
        q_val_5_reg_3372_pp0_iter4_reg <= q_val_5_reg_3372_pp0_iter3_reg;
        q_val_8_reg_3300_pp0_iter4_reg <= q_val_8_reg_3300_pp0_iter3_reg;
        q_val_9_reg_3396_pp0_iter4_reg <= q_val_9_reg_3396_pp0_iter3_reg;
        q_val_reg_3253_pp0_iter4_reg <= q_val_reg_3253_pp0_iter3_reg;
        tmp_10_reg_3487 <= x_2_fu_1726_p3[32'd21];
        tmp_11_reg_3491 <= x_2_fu_1726_p3[32'd20];
        tmp_12_reg_3495 <= x_2_fu_1726_p3[32'd19];
        tmp_13_reg_3499 <= x_2_fu_1726_p3[32'd18];
        tmp_14_reg_3503 <= x_2_fu_1726_p3[32'd17];
        tmp_15_reg_3507 <= x_2_fu_1726_p3[32'd16];
        tmp_16_reg_3511 <= x_2_fu_1726_p3[32'd15];
        tmp_17_reg_3515 <= x_2_fu_1726_p3[32'd14];
        tmp_18_reg_3519 <= x_2_fu_1726_p3[32'd13];
        tmp_19_reg_3523 <= x_2_fu_1726_p3[32'd12];
        tmp_20_reg_3527 <= x_2_fu_1726_p3[32'd11];
        tmp_21_reg_3531 <= x_2_fu_1726_p3[32'd10];
        tmp_22_reg_3535 <= x_2_fu_1726_p3[32'd9];
        tmp_23_reg_3539 <= x_2_fu_1726_p3[32'd8];
        tmp_24_reg_3543 <= x_2_fu_1726_p3[32'd7];
        tmp_25_reg_3547 <= x_2_fu_1726_p3[32'd6];
        tmp_26_reg_3551 <= x_2_fu_1726_p3[32'd5];
        tmp_27_reg_3555 <= x_2_fu_1726_p3[32'd4];
        tmp_28_reg_3559 <= x_2_fu_1726_p3[32'd3];
        tmp_29_reg_3563 <= x_2_fu_1726_p3[32'd2];
        tmp_30_reg_3567 <= x_2_fu_1726_p3[32'd1];
        tmp_9_reg_3483 <= x_2_fu_1726_p3[32'd22];
        trunc_ln10_reg_3478 <= trunc_ln10_fu_1733_p1;
        vs_buf_addr_reg_3472 <= zext_ln169_fu_1711_p1;
        zext_ln169_reg_3458[9 : 0] <= zext_ln169_fu_1711_p1[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
        add_ln146_reg_3440_pp0_iter5_reg <= add_ln146_reg_3440_pp0_iter4_reg;
        addr_cmp_reg_3624 <= addr_cmp_fu_2024_p2;
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        conv_i_i9_i_reg_3614 <= conv_i_i9_i_fu_2015_p2;
        icmp_ln144_reg_3238_pp0_iter5_reg <= icmp_ln144_reg_3238_pp0_iter4_reg;
        q_val_16_reg_3324_pp0_iter5_reg <= q_val_16_reg_3324_pp0_iter4_reg;
        q_val_1_reg_3348_pp0_iter5_reg <= q_val_1_reg_3348_pp0_iter4_reg;
        q_val_2_reg_3414_pp0_iter5_reg <= q_val_2_reg_3414_pp0_iter4_reg;
        q_val_4_reg_3276_pp0_iter5_reg <= q_val_4_reg_3276_pp0_iter4_reg;
        q_val_5_reg_3372_pp0_iter5_reg <= q_val_5_reg_3372_pp0_iter4_reg;
        q_val_8_reg_3300_pp0_iter5_reg <= q_val_8_reg_3300_pp0_iter4_reg;
        q_val_9_reg_3396_pp0_iter5_reg <= q_val_9_reg_3396_pp0_iter4_reg;
        q_val_reg_3253_pp0_iter5_reg <= q_val_reg_3253_pp0_iter4_reg;
        scale_reg_3591 <= scale_fu_1989_p3;
        shl_ln172_1_reg_3629 <= shl_ln172_1_fu_2033_p2;
        shl_ln172_reg_3586_pp0_iter5_reg <= shl_ln172_reg_3586;
        sub_i_i_reg_3597 <= sub_i_i_fu_2001_p2;
        tmp_33_reg_3602 <= sub_i_i_fu_2001_p2[32'd4];
        trunc_ln169_reg_3576_pp0_iter5_reg <= trunc_ln169_reg_3576;
        vs_buf_addr_reg_3472_pp0_iter5_reg <= vs_buf_addr_reg_3472;
        vs_buf_load_reg_3619 <= vs_buf_q0;
        zext_ln169_reg_3458_pp0_iter5_reg[9 : 0] <= zext_ln169_reg_3458[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
        icmp_ln144_reg_3238_pp0_iter6_reg <= icmp_ln144_reg_3238_pp0_iter5_reg;
        scale_reg_3591_pp0_iter6_reg <= scale_reg_3591;
        shl_ln169_reg_3638_pp0_iter6_reg[5 : 3] <= shl_ln169_reg_3638[5 : 3];
        trunc_ln169_reg_3576_pp0_iter6_reg <= trunc_ln169_reg_3576_pp0_iter5_reg;
        zext_ln169_reg_3458_pp0_iter6_reg[9 : 0] <= zext_ln169_reg_3458_pp0_iter5_reg[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
        ashr_ln163_1_reg_3658 <= ashr_ln163_1_fu_2089_p2;
        ashr_ln163_2_reg_3668 <= ashr_ln163_2_fu_2107_p2;
        ashr_ln163_3_reg_3678 <= ashr_ln163_3_fu_2125_p2;
        ashr_ln163_4_reg_3688 <= ashr_ln163_4_fu_2143_p2;
        ashr_ln163_5_reg_3698 <= ashr_ln163_5_fu_2161_p2;
        ashr_ln163_6_reg_3708 <= ashr_ln163_6_fu_2179_p2;
        ashr_ln163_7_reg_3718 <= ashr_ln163_7_fu_2197_p2;
        ashr_ln163_reg_3648 <= ashr_ln163_fu_2071_p2;
        lnot_i_i_reg_3634 <= lnot_i_i_fu_2042_p2;
        shl_ln163_1_reg_3653 <= shl_ln163_1_fu_2080_p2;
        shl_ln163_2_reg_3663 <= shl_ln163_2_fu_2098_p2;
        shl_ln163_3_reg_3673 <= shl_ln163_3_fu_2116_p2;
        shl_ln163_4_reg_3683 <= shl_ln163_4_fu_2134_p2;
        shl_ln163_5_reg_3693 <= shl_ln163_5_fu_2152_p2;
        shl_ln163_6_reg_3703 <= shl_ln163_6_fu_2170_p2;
        shl_ln163_7_reg_3713 <= shl_ln163_7_fu_2188_p2;
        shl_ln163_reg_3643 <= shl_ln163_fu_2062_p2;
        shl_ln169_reg_3638[5 : 3] <= shl_ln169_fu_2053_p2[5 : 3];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        cp_1_reg_3233 <= ap_sig_allocacmp_cp_1;
        icmp_ln144_reg_3238 <= icmp_ln144_fu_1146_p2;
        mul_reg_3228[5 : 3] <= mul_fu_1120_p3[5 : 3];
        q_val_reg_3253 <= q_val_fu_1162_p19;
        sub_ln150_reg_3261 <= sub_ln150_fu_1202_p2;
        tmp_reg_3266 <= q_val_fu_1162_p19[32'd22];
        trunc_ln144_reg_3242 <= trunc_ln144_fu_1158_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter7_fsm_state16))) begin
        icmp_ln144_reg_3238_pp0_iter7_reg <= icmp_ln144_reg_3238_pp0_iter6_reg;
        shl_ln169_1_reg_3811 <= shl_ln169_1_fu_2823_p2;
        shl_ln169_2_reg_3816 <= shl_ln169_2_fu_2832_p2;
        shl_ln169_3_reg_3828 <= shl_ln169_3_fu_2841_p2;
        shl_ln169_4_reg_3833 <= shl_ln169_4_fu_2850_p2;
        shl_ln169_5_reg_3838 <= shl_ln169_5_fu_2859_p2;
        shl_ln169_6_reg_3843 <= shl_ln169_6_fu_2868_p2;
        shl_ln169_7_reg_3848 <= shl_ln169_7_fu_2877_p2;
        shl_ln169_8_reg_3853 <= shl_ln169_8_fu_2886_p2;
        shl_ln169_9_reg_3858 <= shl_ln169_9_fu_2895_p2;
        zext_ln169_reg_3458_pp0_iter7_reg[9 : 0] <= zext_ln169_reg_3458_pp0_iter6_reg[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter7_fsm_state15)) begin
        q_out_10_reg_3775 <= q_out_10_fu_2569_p3;
        q_out_11_reg_3781 <= q_out_11_fu_2617_p3;
        q_out_12_reg_3787 <= q_out_12_fu_2665_p3;
        q_out_13_reg_3793 <= q_out_13_fu_2713_p3;
        q_out_14_reg_3799 <= q_out_14_fu_2761_p3;
        q_out_8_reg_3763 <= q_out_8_fu_2473_p3;
        q_out_9_reg_3769 <= q_out_9_fu_2521_p3;
        q_out_reg_3805 <= q_out_fu_2809_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state11))) begin
        shl_ln172_reg_3586 <= shl_ln172_fu_1935_p2;
        trunc_ln169_reg_3576 <= trunc_ln169_fu_1920_p1;
        zext_ln172_reg_3581[4 : 2] <= zext_ln172_fu_1931_p1[4 : 2];
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

assign ap_ST_iter2_fsm_state5_blk = 1'b0;

assign ap_ST_iter2_fsm_state6_blk = 1'b0;

assign ap_ST_iter3_fsm_state7_blk = 1'b0;

assign ap_ST_iter3_fsm_state8_blk = 1'b0;

assign ap_ST_iter4_fsm_state10_blk = 1'b0;

assign ap_ST_iter4_fsm_state9_blk = 1'b0;

assign ap_ST_iter5_fsm_state11_blk = 1'b0;

assign ap_ST_iter5_fsm_state12_blk = 1'b0;

assign ap_ST_iter6_fsm_state13_blk = 1'b0;

assign ap_ST_iter6_fsm_state14_blk = 1'b0;

assign ap_ST_iter7_fsm_state15_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) begin
        ap_ST_iter7_fsm_state16_blk = 1'b1;
    end else begin
        ap_ST_iter7_fsm_state16_blk = 1'b0;
    end
end

assign ap_ST_iter8_fsm_state17_blk = 1'b0;

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln144_reg_3238 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (ap_loop_exit_ready_pp0_iter8_reg == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state0) & (1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_cp_1 = 4'd0;
    end else begin
        ap_sig_allocacmp_cp_1 = cp_fu_388;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_0_ce1 = 1'b1;
    end else begin
        vq_buf_0_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_0_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_0_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_1_ce1 = 1'b1;
    end else begin
        vq_buf_1_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_1_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_1_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_2_ce1 = 1'b1;
    end else begin
        vq_buf_2_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_2_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_2_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_3_ce1 = 1'b1;
    end else begin
        vq_buf_3_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_3_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_3_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_4_ce1 = 1'b1;
    end else begin
        vq_buf_4_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_4_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_4_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_5_ce1 = 1'b1;
    end else begin
        vq_buf_5_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_5_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_5_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_6_ce1 = 1'b1;
    end else begin
        vq_buf_6_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_6_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_6_we1 = 8'd0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter8_fsm_state17)) begin
        vq_buf_7_ce1 = 1'b1;
    end else begin
        vq_buf_7_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd0))) begin
        vq_buf_7_we1 = shl_ln169_2_reg_3816;
    end else begin
        vq_buf_7_we1 = 8'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state16) & (ap_predicate_op408_write_state16 == 1'b1))) begin
        vq_cache_o_stream_TDATA_blk_n = vq_cache_o_stream_TREADY;
    end else begin
        vq_cache_o_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter7_fsm_state16) & (ap_predicate_op408_write_state16 == 1'b1))) begin
        vq_cache_o_stream_TVALID = 1'b1;
    end else begin
        vq_cache_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state12)) | (~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state11)) | (~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter4_fsm_state10)))) begin
        vs_buf_ce0 = 1'b1;
    end else begin
        vs_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
        vs_buf_ce1 = 1'b1;
    end else begin
        vs_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0))) begin
        vs_buf_we1 = 1'b1;
    end else begin
        vs_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state16) & (ap_predicate_op410_write_state16 == 1'b1))) begin
        vs_cache_o_stream_TDATA_blk_n = vs_cache_o_stream_TREADY;
    end else begin
        vs_cache_o_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter7_fsm_state16) & (ap_predicate_op410_write_state16 == 1'b1))) begin
        vs_cache_o_stream_TVALID = 1'b1;
    end else begin
        vs_cache_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if (~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)))) begin
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
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end
        end
        ap_ST_iter1_fsm_state4 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b0 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
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
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end
        end
        ap_ST_iter2_fsm_state6 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b0 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
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
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end
        end
        ap_ST_iter3_fsm_state8 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b0 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
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
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end
        end
        ap_ST_iter4_fsm_state10 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b0 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
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
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state11))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state12;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end
        end
        ap_ST_iter5_fsm_state12 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b0 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state12;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
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
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state14;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end
        end
        ap_ST_iter6_fsm_state14 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b0 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state14;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
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
            ap_NS_iter7_fsm = ap_ST_iter7_fsm_state16;
        end
        ap_ST_iter7_fsm_state16 : begin
            if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end else if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b0 == ap_CS_iter6_fsm_state14))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state16;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
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

always @ (*) begin
    case (ap_CS_iter8_fsm)
        ap_ST_iter8_fsm_state17 : begin
            if (((1'b0 == ap_CS_iter7_fsm_state16) | ((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state0;
            end else if (((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter7_fsm_state16)) | ((1'b1 == ap_CS_iter8_fsm_state17) & (icmp_ln144_reg_3238_pp0_iter7_reg == 1'd1)))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state17;
            end else begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state17;
            end
        end
        ap_ST_iter8_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)) & (1'b1 == ap_CS_iter7_fsm_state16))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state17;
            end else begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter8_fsm = 'bx;
        end
    endcase
end

assign abs_max_1_fu_1303_p3 = ((icmp_ln224_1_fu_1297_p2[0:0] == 1'b1) ? select_ln150_2_fu_1292_p3 : zext_ln147_fu_1289_p1);

assign abs_max_2_fu_1366_p3 = ((icmp_ln224_2_fu_1361_p2[0:0] == 1'b1) ? select_ln150_4_fu_1356_p3 : abs_max_1_reg_3294);

assign abs_max_3_fu_1428_p3 = ((icmp_ln224_3_fu_1423_p2[0:0] == 1'b1) ? select_ln150_6_fu_1418_p3 : abs_max_2_reg_3318);

assign abs_max_4_fu_1490_p3 = ((icmp_ln224_4_fu_1485_p2[0:0] == 1'b1) ? select_ln150_8_fu_1480_p3 : abs_max_3_reg_3342);

assign abs_max_5_fu_1552_p3 = ((icmp_ln224_5_fu_1547_p2[0:0] == 1'b1) ? select_ln150_10_fu_1542_p3 : abs_max_4_reg_3366);

assign abs_max_6_fu_1653_p3 = ((icmp_ln224_6_fu_1648_p2[0:0] == 1'b1) ? select_ln150_12_fu_1643_p3 : abs_max_5_reg_3390);

assign abs_max_8_fu_1693_p3 = ((icmp_ln224_7_fu_1689_p2[0:0] == 1'b1) ? select_ln150_14_reg_3434 : abs_max_6_reg_3428);

assign abs_max_fu_1236_p3 = ((icmp_ln224_fu_1230_p2[0:0] == 1'b1) ? trunc_ln224_fu_1226_p1 : 22'd0);

assign add_ln144_fu_1152_p2 = (ap_sig_allocacmp_cp_1 + 4'd1);

assign add_ln146_fu_1674_p2 = (zext_ln144_fu_1671_p1 + mul_reg_3228);

assign add_ln169_fu_1706_p2 = (tmp_s_fu_1699_p3 + st);

assign addr_cmp_fu_2024_p2 = ((reuse_addr_reg_fu_380 == zext_ln169_reg_3458) ? 1'b1 : 1'b0);

assign and_ln172_fu_2216_p2 = (xor_ln172_fu_2211_p2 & reuse_select_fu_2205_p3);

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

assign ap_CS_iter7_fsm_state16 = ap_CS_iter7_fsm[32'd2];

assign ap_CS_iter8_fsm_state0 = ap_CS_iter8_fsm[32'd0];

assign ap_CS_iter8_fsm_state17 = ap_CS_iter8_fsm[32'd1];

always @ (*) begin
    ap_block_state16_io = (((ap_predicate_op410_write_state16 == 1'b1) & (vs_cache_o_stream_TREADY == 1'b0)) | ((ap_predicate_op408_write_state16 == 1'b1) & (vq_cache_o_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state16_pp0_stage1_iter7 = (((ap_predicate_op410_write_state16 == 1'b1) & (vs_cache_o_stream_TREADY == 1'b0)) | ((ap_predicate_op408_write_state16 == 1'b1) & (vq_cache_o_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_condition_192 = ~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)));
end

always @ (*) begin
    ap_condition_2652 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd1) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2657 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd1) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2663 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd1) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2670 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd1) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2678 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd1) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2687 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd1) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2697 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd1) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2708 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd1) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2720 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd1) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2733 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd1) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2747 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd1) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2762 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd1) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2778 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd1) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2795 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd1) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2813 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd1) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2832 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd1) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2852 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd1) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2873 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_26_reg_3551_pp0_iter4_reg == 1'd1) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd0) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2895 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_27_reg_3555_pp0_iter4_reg == 1'd1) & (tmp_26_reg_3551_pp0_iter4_reg == 1'd0) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd0) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2918 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_28_reg_3559_pp0_iter4_reg == 1'd1) & (tmp_27_reg_3555_pp0_iter4_reg == 1'd0) & (tmp_26_reg_3551_pp0_iter4_reg == 1'd0) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd0) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2942 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_29_reg_3563_pp0_iter4_reg == 1'd1) & (tmp_28_reg_3559_pp0_iter4_reg == 1'd0) & (tmp_27_reg_3555_pp0_iter4_reg == 1'd0) & (tmp_26_reg_3551_pp0_iter4_reg == 1'd0) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd0) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2967 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_30_reg_3567_pp0_iter4_reg == 1'd1) & (tmp_29_reg_3563_pp0_iter4_reg == 1'd0) & (tmp_28_reg_3559_pp0_iter4_reg == 1'd0) & (tmp_27_reg_3555_pp0_iter4_reg == 1'd0) & (tmp_26_reg_3551_pp0_iter4_reg == 1'd0) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd0) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2992 = ((1'b1 == ap_CS_iter5_fsm_state11) & (tmp_30_reg_3567_pp0_iter4_reg == 1'd0) & (tmp_29_reg_3563_pp0_iter4_reg == 1'd0) & (tmp_28_reg_3559_pp0_iter4_reg == 1'd0) & (tmp_27_reg_3555_pp0_iter4_reg == 1'd0) & (tmp_26_reg_3551_pp0_iter4_reg == 1'd0) & (tmp_25_reg_3547_pp0_iter4_reg == 1'd0) & (tmp_24_reg_3543_pp0_iter4_reg == 1'd0) & (tmp_23_reg_3539_pp0_iter4_reg == 1'd0) & (tmp_22_reg_3535_pp0_iter4_reg == 1'd0) & (tmp_21_reg_3531_pp0_iter4_reg == 1'd0) & (tmp_20_reg_3527_pp0_iter4_reg == 1'd0) & (tmp_19_reg_3523_pp0_iter4_reg == 1'd0) & (tmp_18_reg_3519_pp0_iter4_reg == 1'd0) & (tmp_17_reg_3515_pp0_iter4_reg == 1'd0) & (tmp_16_reg_3511_pp0_iter4_reg == 1'd0) & (tmp_15_reg_3507_pp0_iter4_reg == 1'd0) & (tmp_14_reg_3503_pp0_iter4_reg == 1'd0) & (tmp_13_reg_3499_pp0_iter4_reg == 1'd0) & (tmp_12_reg_3495_pp0_iter4_reg == 1'd0) & (tmp_11_reg_3491_pp0_iter4_reg == 1'd0) & (tmp_10_reg_3487_pp0_iter4_reg == 1'd0) & (tmp_9_reg_3483_pp0_iter4_reg == 1'd0) & (icmp_ln144_reg_3238_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_2998 = ((1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln144_reg_3238_pp0_iter5_reg == 1'd0) & (lnot_i_i_fu_2042_p2 == 1'd1));
end

always @ (*) begin
    ap_condition_385 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_394 = (~((1'b1 == ap_CS_iter7_fsm_state16) & ((1'b1 == ap_block_state16_io) | (1'b1 == ap_block_state16_pp0_stage1_iter7))) & (1'b1 == ap_CS_iter6_fsm_state14));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage1;

assign ap_phi_reg_pp0_iter0_q_val_0137139145151161171185_reg_1075 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_1141143153159173183_reg_1084 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_2147149163169187_reg_1066 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_3155157175181_reg_1093 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_4165167189_reg_1057 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_5177179_reg_1102 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_6191_reg_1048 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_32_reg_1111 = 'bx;

assign ap_phi_reg_pp0_iter0_s_val_reg_953 = 'bx;

always @ (*) begin
    ap_predicate_op408_write_state16 = ((write_cache_stream == 1'd1) & (icmp_ln144_reg_3238_pp0_iter6_reg == 1'd0));
end

always @ (*) begin
    ap_predicate_op410_write_state16 = ((write_cache_stream == 1'd1) & (icmp_ln144_reg_3238_pp0_iter6_reg == 1'd0));
end

assign ashr_ln163_1_fu_2089_p2 = $signed(q_val_4_reg_3276_pp0_iter5_reg) >>> conv_i_i_i41cast385_fu_2085_p1;

assign ashr_ln163_2_fu_2107_p2 = $signed(q_val_8_reg_3300_pp0_iter5_reg) >>> conv_i_i_i41cast387_fu_2103_p1;

assign ashr_ln163_3_fu_2125_p2 = $signed(q_val_16_reg_3324_pp0_iter5_reg) >>> conv_i_i_i41cast389_fu_2121_p1;

assign ashr_ln163_4_fu_2143_p2 = $signed(q_val_1_reg_3348_pp0_iter5_reg) >>> conv_i_i_i41cast391_fu_2139_p1;

assign ashr_ln163_5_fu_2161_p2 = $signed(q_val_5_reg_3372_pp0_iter5_reg) >>> conv_i_i_i41cast393_fu_2157_p1;

assign ashr_ln163_6_fu_2179_p2 = $signed(q_val_9_reg_3396_pp0_iter5_reg) >>> conv_i_i_i41cast395_fu_2175_p1;

assign ashr_ln163_7_fu_2197_p2 = $signed(q_val_2_reg_3414_pp0_iter5_reg) >>> conv_i_i_i41cast397_fu_2193_p1;

assign ashr_ln163_fu_2071_p2 = $signed(q_val_reg_3253_pp0_iter5_reg) >>> conv_i_i_i41cast_fu_2067_p1;

assign conv_i_i9_i_cast_fu_2047_p1 = $signed(conv_i_i9_i_reg_3614);

assign conv_i_i9_i_castcast384_fu_2076_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast386_fu_2094_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast388_fu_2112_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast390_fu_2130_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast392_fu_2148_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast394_fu_2166_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast396_fu_2184_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_castcast_fu_2058_p1 = conv_i_i9_i_cast_fu_2047_p1[22:0];

assign conv_i_i9_i_fu_2015_p2 = (5'd1 - scale_cast_fu_1997_p1);

assign conv_i_i_i41_fu_2050_p1 = sub_i_i_reg_3597;

assign conv_i_i_i41cast385_fu_2085_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast387_fu_2103_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast389_fu_2121_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast391_fu_2139_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast393_fu_2157_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast395_fu_2175_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast397_fu_2193_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign conv_i_i_i41cast_fu_2067_p1 = conv_i_i_i41_fu_2050_p1[22:0];

assign icmp_ln12_fu_1716_p2 = (($signed(abs_max_8_reg_3451) > $signed(23'd0)) ? 1'b1 : 1'b0);

assign icmp_ln144_fu_1146_p2 = ((ap_sig_allocacmp_cp_1 == 4'd8) ? 1'b1 : 1'b0);

assign icmp_ln224_1_fu_1297_p2 = (($signed(zext_ln147_fu_1289_p1) < $signed(select_ln150_2_fu_1292_p3)) ? 1'b1 : 1'b0);

assign icmp_ln224_2_fu_1361_p2 = (($signed(abs_max_1_reg_3294) < $signed(select_ln150_4_fu_1356_p3)) ? 1'b1 : 1'b0);

assign icmp_ln224_3_fu_1423_p2 = (($signed(abs_max_2_reg_3318) < $signed(select_ln150_6_fu_1418_p3)) ? 1'b1 : 1'b0);

assign icmp_ln224_4_fu_1485_p2 = (($signed(abs_max_3_reg_3342) < $signed(select_ln150_8_fu_1480_p3)) ? 1'b1 : 1'b0);

assign icmp_ln224_5_fu_1547_p2 = (($signed(abs_max_4_reg_3366) < $signed(select_ln150_10_fu_1542_p3)) ? 1'b1 : 1'b0);

assign icmp_ln224_6_fu_1648_p2 = (($signed(abs_max_5_reg_3390) < $signed(select_ln150_12_fu_1643_p3)) ? 1'b1 : 1'b0);

assign icmp_ln224_7_fu_1689_p2 = (($signed(abs_max_6_reg_3428) < $signed(select_ln150_14_reg_3434)) ? 1'b1 : 1'b0);

assign icmp_ln224_fu_1230_p2 = (($signed(select_ln150_fu_1221_p3) > $signed(23'd0)) ? 1'b1 : 1'b0);

assign icmp_ln42_1_fu_2481_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0_1141143153159173183_reg_1084) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_2_fu_2529_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0_2147149163169187_reg_1066) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_3_fu_2577_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0_3155157175181_reg_1093) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_4_fu_2625_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0_4165167189_reg_1057) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_5_fu_2673_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0_5177179_reg_1102) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_6_fu_2721_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0_6191_reg_1048) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_7_fu_2769_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_32_reg_1111) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_fu_2433_p2 = (($signed(ap_phi_reg_pp0_iter7_q_val_0137139145151161171185_reg_1075) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln43_1_fu_2449_p2 = (($signed(tmp_34_fu_2439_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_2_fu_2497_p2 = (($signed(tmp_35_fu_2487_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_3_fu_2545_p2 = (($signed(tmp_36_fu_2535_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_4_fu_2593_p2 = (($signed(tmp_37_fu_2583_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_5_fu_2641_p2 = (($signed(tmp_38_fu_2631_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_6_fu_2689_p2 = (($signed(tmp_39_fu_2679_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_7_fu_2737_p2 = (($signed(tmp_40_fu_2727_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_8_fu_2785_p2 = (($signed(tmp_41_fu_2775_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_fu_1959_p2 = ((tmp_32_fu_1949_p4 == 2'd1) ? 1'b1 : 1'b0);

assign lnot_i_i_fu_2042_p2 = ((scale_reg_3591 == 4'd0) ? 1'b1 : 1'b0);

assign mul_fu_1120_p3 = {{hct}, {3'd0}};

assign or_ln154_fu_1983_p2 = (tmp_31_fu_1941_p3 | icmp_ln43_fu_1959_p2);

assign or_ln168_1_fu_2515_p2 = (icmp_ln43_2_fu_2497_p2 | icmp_ln42_1_fu_2481_p2);

assign or_ln168_2_fu_2563_p2 = (icmp_ln43_3_fu_2545_p2 | icmp_ln42_2_fu_2529_p2);

assign or_ln168_3_fu_2611_p2 = (icmp_ln43_4_fu_2593_p2 | icmp_ln42_3_fu_2577_p2);

assign or_ln168_4_fu_2659_p2 = (icmp_ln43_5_fu_2641_p2 | icmp_ln42_4_fu_2625_p2);

assign or_ln168_5_fu_2707_p2 = (icmp_ln43_6_fu_2689_p2 | icmp_ln42_5_fu_2673_p2);

assign or_ln168_6_fu_2755_p2 = (icmp_ln43_7_fu_2737_p2 | icmp_ln42_6_fu_2721_p2);

assign or_ln168_7_fu_2803_p2 = (icmp_ln43_8_fu_2785_p2 | icmp_ln42_7_fu_2769_p2);

assign or_ln168_fu_2467_p2 = (icmp_ln43_1_fu_2449_p2 | icmp_ln42_fu_2433_p2);

assign or_ln172_fu_2222_p2 = (shl_ln172_1_reg_3629 | and_ln172_fu_2216_p2);

assign q_out_10_fu_2569_p3 = ((or_ln168_2_fu_2563_p2[0:0] == 1'b1) ? select_ln168_4_fu_2555_p3 : trunc_ln168_2_fu_2551_p1);

assign q_out_11_fu_2617_p3 = ((or_ln168_3_fu_2611_p2[0:0] == 1'b1) ? select_ln168_6_fu_2603_p3 : trunc_ln168_3_fu_2599_p1);

assign q_out_12_fu_2665_p3 = ((or_ln168_4_fu_2659_p2[0:0] == 1'b1) ? select_ln168_8_fu_2651_p3 : trunc_ln168_4_fu_2647_p1);

assign q_out_13_fu_2713_p3 = ((or_ln168_5_fu_2707_p2[0:0] == 1'b1) ? select_ln168_10_fu_2699_p3 : trunc_ln168_5_fu_2695_p1);

assign q_out_14_fu_2761_p3 = ((or_ln168_6_fu_2755_p2[0:0] == 1'b1) ? select_ln168_12_fu_2747_p3 : trunc_ln168_6_fu_2743_p1);

assign q_out_8_fu_2473_p3 = ((or_ln168_fu_2467_p2[0:0] == 1'b1) ? select_ln168_fu_2459_p3 : trunc_ln168_fu_2455_p1);

assign q_out_9_fu_2521_p3 = ((or_ln168_1_fu_2515_p2[0:0] == 1'b1) ? select_ln168_2_fu_2507_p3 : trunc_ln168_1_fu_2503_p1);

assign q_out_fu_2809_p3 = ((or_ln168_7_fu_2803_p2[0:0] == 1'b1) ? select_ln168_14_fu_2795_p3 : trunc_ln168_7_fu_2791_p1);

assign q_val_10_fu_2258_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_1_reg_3653 : ashr_ln163_1_reg_3658);

assign q_val_11_fu_2263_p2 = (q_val_10_fu_2258_p3 + 23'd1);

assign q_val_12_fu_2269_p4 = {{q_val_11_fu_2263_p2[22:1]}};

assign q_val_13_fu_2283_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_2_reg_3663 : ashr_ln163_2_reg_3668);

assign q_val_14_fu_2288_p2 = (q_val_13_fu_2283_p3 + 23'd1);

assign q_val_15_fu_2294_p4 = {{q_val_14_fu_2288_p2[22:1]}};

assign q_val_16_fu_1373_p17 = 'bx;

assign q_val_17_fu_2308_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_3_reg_3673 : ashr_ln163_3_reg_3678);

assign q_val_18_fu_2313_p2 = (q_val_17_fu_2308_p3 + 23'd1);

assign q_val_19_fu_2319_p4 = {{q_val_18_fu_2313_p2[22:1]}};

assign q_val_1_fu_1435_p17 = 'bx;

assign q_val_20_fu_2333_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_4_reg_3683 : ashr_ln163_4_reg_3688);

assign q_val_21_fu_2338_p2 = (q_val_20_fu_2333_p3 + 23'd1);

assign q_val_22_fu_2344_p4 = {{q_val_21_fu_2338_p2[22:1]}};

assign q_val_23_fu_2358_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_5_reg_3693 : ashr_ln163_5_reg_3698);

assign q_val_24_fu_2363_p2 = (q_val_23_fu_2358_p3 + 23'd1);

assign q_val_25_fu_2369_p4 = {{q_val_24_fu_2363_p2[22:1]}};

assign q_val_26_fu_2383_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_6_reg_3703 : ashr_ln163_6_reg_3708);

assign q_val_27_fu_2388_p2 = (q_val_26_fu_2383_p3 + 23'd1);

assign q_val_28_fu_2394_p4 = {{q_val_27_fu_2388_p2[22:1]}};

assign q_val_29_fu_2408_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_7_reg_3713 : ashr_ln163_7_reg_3718);

assign q_val_2_fu_1604_p17 = 'bx;

assign q_val_30_fu_2413_p2 = (q_val_29_fu_2408_p3 + 23'd1);

assign q_val_31_fu_2419_p4 = {{q_val_30_fu_2413_p2[22:1]}};

assign q_val_3_fu_2233_p3 = ((tmp_33_reg_3602[0:0] == 1'b1) ? shl_ln163_reg_3643 : ashr_ln163_reg_3648);

assign q_val_4_fu_1244_p17 = 'bx;

assign q_val_5_fu_1497_p17 = 'bx;

assign q_val_6_fu_2238_p2 = (q_val_3_fu_2233_p3 + 23'd1);

assign q_val_7_fu_2244_p4 = {{q_val_6_fu_2238_p2[22:1]}};

assign q_val_8_fu_1311_p17 = 'bx;

assign q_val_9_fu_1559_p17 = 'bx;

assign q_val_fu_1162_p17 = 'bx;

assign q_val_fu_1162_p18 = ap_sig_allocacmp_cp_1[2:0];

assign reuse_select_fu_2205_p3 = ((addr_cmp_reg_3624[0:0] == 1'b1) ? reuse_reg_fu_384 : vs_buf_load_reg_3619);

assign scale_cast_fu_1997_p1 = scale_fu_1989_p3;

assign scale_fu_1989_p3 = ((or_ln154_fu_1983_p2[0:0] == 1'b1) ? select_ln154_fu_1975_p3 : trunc_ln154_fu_1965_p1);

assign select_ln150_10_fu_1542_p3 = ((tmp_6_reg_3385[0:0] == 1'b1) ? sub_ln150_5_reg_3380 : q_val_5_reg_3372);

assign select_ln150_12_fu_1643_p3 = ((tmp_7_reg_3409[0:0] == 1'b1) ? sub_ln150_6_reg_3404 : q_val_9_reg_3396);

assign select_ln150_14_fu_1665_p3 = ((tmp_8_reg_3423[0:0] == 1'b1) ? sub_ln150_7_fu_1660_p2 : q_val_2_reg_3414);

assign select_ln150_2_fu_1292_p3 = ((tmp_2_reg_3289[0:0] == 1'b1) ? sub_ln150_1_reg_3284 : q_val_4_reg_3276);

assign select_ln150_4_fu_1356_p3 = ((tmp_3_reg_3313[0:0] == 1'b1) ? sub_ln150_2_reg_3308 : q_val_8_reg_3300);

assign select_ln150_6_fu_1418_p3 = ((tmp_4_reg_3337[0:0] == 1'b1) ? sub_ln150_3_reg_3332 : q_val_16_reg_3324);

assign select_ln150_8_fu_1480_p3 = ((tmp_5_reg_3361[0:0] == 1'b1) ? sub_ln150_4_reg_3356 : q_val_1_reg_3348);

assign select_ln150_fu_1221_p3 = ((tmp_reg_3266[0:0] == 1'b1) ? sub_ln150_reg_3261 : q_val_reg_3253);

assign select_ln154_fu_1975_p3 = ((xor_ln154_fu_1969_p2[0:0] == 1'b1) ? 4'd15 : 4'd0);

assign select_ln168_10_fu_2699_p3 = ((icmp_ln42_5_fu_2673_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_12_fu_2747_p3 = ((icmp_ln42_6_fu_2721_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_14_fu_2795_p3 = ((icmp_ln42_7_fu_2769_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_2_fu_2507_p3 = ((icmp_ln42_1_fu_2481_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_4_fu_2555_p3 = ((icmp_ln42_2_fu_2529_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_6_fu_2603_p3 = ((icmp_ln42_3_fu_2577_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_8_fu_2651_p3 = ((icmp_ln42_4_fu_2625_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln168_fu_2459_p3 = ((icmp_ln42_fu_2433_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln16_fu_1913_p3 = ((trunc_ln10_reg_3478[0:0] == 1'b1) ? 6'd58 : 6'd57);

assign sext_ln165_1_fu_2279_p1 = $signed(q_val_12_fu_2269_p4);

assign sext_ln165_2_fu_2304_p1 = $signed(q_val_15_fu_2294_p4);

assign sext_ln165_3_fu_2329_p1 = $signed(q_val_19_fu_2319_p4);

assign sext_ln165_4_fu_2354_p1 = $signed(q_val_22_fu_2344_p4);

assign sext_ln165_5_fu_2379_p1 = $signed(q_val_25_fu_2369_p4);

assign sext_ln165_6_fu_2404_p1 = $signed(q_val_28_fu_2394_p4);

assign sext_ln165_7_fu_2429_p1 = $signed(q_val_31_fu_2419_p4);

assign sext_ln165_fu_2254_p1 = $signed(q_val_7_fu_2244_p4);

assign shl_ln163_1_fu_2080_p2 = q_val_4_reg_3276_pp0_iter5_reg << conv_i_i9_i_castcast384_fu_2076_p1;

assign shl_ln163_2_fu_2098_p2 = q_val_8_reg_3300_pp0_iter5_reg << conv_i_i9_i_castcast386_fu_2094_p1;

assign shl_ln163_3_fu_2116_p2 = q_val_16_reg_3324_pp0_iter5_reg << conv_i_i9_i_castcast388_fu_2112_p1;

assign shl_ln163_4_fu_2134_p2 = q_val_1_reg_3348_pp0_iter5_reg << conv_i_i9_i_castcast390_fu_2130_p1;

assign shl_ln163_5_fu_2152_p2 = q_val_5_reg_3372_pp0_iter5_reg << conv_i_i9_i_castcast392_fu_2148_p1;

assign shl_ln163_6_fu_2170_p2 = q_val_9_reg_3396_pp0_iter5_reg << conv_i_i9_i_castcast394_fu_2166_p1;

assign shl_ln163_7_fu_2188_p2 = q_val_2_reg_3414_pp0_iter5_reg << conv_i_i9_i_castcast396_fu_2184_p1;

assign shl_ln163_fu_2062_p2 = q_val_reg_3253_pp0_iter5_reg << conv_i_i9_i_castcast_fu_2058_p1;

assign shl_ln169_1_fu_2823_p2 = zext_ln169_2_fu_2820_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_2_fu_2832_p2 = 8'd1 << zext_ln169_3_fu_2829_p1;

assign shl_ln169_3_fu_2841_p2 = zext_ln169_4_fu_2838_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_4_fu_2850_p2 = zext_ln169_5_fu_2847_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_5_fu_2859_p2 = zext_ln169_6_fu_2856_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_6_fu_2868_p2 = zext_ln169_7_fu_2865_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_7_fu_2877_p2 = zext_ln169_8_fu_2874_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_8_fu_2886_p2 = zext_ln169_9_fu_2883_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_9_fu_2895_p2 = zext_ln169_10_fu_2892_p1 << zext_ln169_1_fu_2817_p1;

assign shl_ln169_fu_2053_p2 = add_ln146_reg_3440_pp0_iter5_reg << 6'd3;

assign shl_ln172_1_fu_2033_p2 = zext_ln172_1_fu_2029_p1 << zext_ln172_reg_3581;

assign shl_ln172_fu_1935_p2 = 32'd15 << zext_ln172_fu_1931_p1;

assign shl_ln1_fu_1923_p3 = {{trunc_ln169_fu_1920_p1}, {2'd0}};

assign sub_i_i_fu_2001_p2 = ($signed(scale_cast_fu_1997_p1) + $signed(5'd31));

assign sub_ln150_1_fu_1275_p2 = (23'd0 - q_val_4_fu_1244_p19);

assign sub_ln150_2_fu_1342_p2 = (23'd0 - q_val_8_fu_1311_p19);

assign sub_ln150_3_fu_1404_p2 = (23'd0 - q_val_16_fu_1373_p19);

assign sub_ln150_4_fu_1466_p2 = (23'd0 - q_val_1_fu_1435_p19);

assign sub_ln150_5_fu_1528_p2 = (23'd0 - q_val_5_fu_1497_p19);

assign sub_ln150_6_fu_1590_p2 = (23'd0 - q_val_9_fu_1559_p19);

assign sub_ln150_7_fu_1660_p2 = (23'd0 - q_val_2_reg_3414);

assign sub_ln150_fu_1202_p2 = (23'd0 - q_val_fu_1162_p19);

assign tmp_10_reg_3487_pp0_iter4_reg = tmp_10_reg_3487;

assign tmp_11_reg_3491_pp0_iter4_reg = tmp_11_reg_3491;

assign tmp_12_reg_3495_pp0_iter4_reg = tmp_12_reg_3495;

assign tmp_13_reg_3499_pp0_iter4_reg = tmp_13_reg_3499;

assign tmp_14_reg_3503_pp0_iter4_reg = tmp_14_reg_3503;

assign tmp_15_reg_3507_pp0_iter4_reg = tmp_15_reg_3507;

assign tmp_16_reg_3511_pp0_iter4_reg = tmp_16_reg_3511;

assign tmp_17_reg_3515_pp0_iter4_reg = tmp_17_reg_3515;

assign tmp_18_reg_3519_pp0_iter4_reg = tmp_18_reg_3519;

assign tmp_19_reg_3523_pp0_iter4_reg = tmp_19_reg_3523;

assign tmp_20_reg_3527_pp0_iter4_reg = tmp_20_reg_3527;

assign tmp_21_reg_3531_pp0_iter4_reg = tmp_21_reg_3531;

assign tmp_22_reg_3535_pp0_iter4_reg = tmp_22_reg_3535;

assign tmp_23_reg_3539_pp0_iter4_reg = tmp_23_reg_3539;

assign tmp_24_reg_3543_pp0_iter4_reg = tmp_24_reg_3543;

assign tmp_25_reg_3547_pp0_iter4_reg = tmp_25_reg_3547;

assign tmp_26_reg_3551_pp0_iter4_reg = tmp_26_reg_3551;

assign tmp_27_reg_3555_pp0_iter4_reg = tmp_27_reg_3555;

assign tmp_28_reg_3559_pp0_iter4_reg = tmp_28_reg_3559;

assign tmp_29_reg_3563_pp0_iter4_reg = tmp_29_reg_3563;

assign tmp_30_reg_3567_pp0_iter4_reg = tmp_30_reg_3567;

assign tmp_31_fu_1941_p3 = ap_phi_reg_pp0_iter5_s_val_reg_953[32'd5];

assign tmp_32_fu_1949_p4 = {{ap_phi_reg_pp0_iter5_s_val_reg_953[5:4]}};

assign tmp_34_fu_2439_p4 = {{ap_phi_reg_pp0_iter7_q_val_0137139145151161171185_reg_1075[22:7]}};

assign tmp_35_fu_2487_p4 = {{ap_phi_reg_pp0_iter7_q_val_0_1141143153159173183_reg_1084[22:7]}};

assign tmp_36_fu_2535_p4 = {{ap_phi_reg_pp0_iter7_q_val_0_2147149163169187_reg_1066[22:7]}};

assign tmp_37_fu_2583_p4 = {{ap_phi_reg_pp0_iter7_q_val_0_3155157175181_reg_1093[22:7]}};

assign tmp_38_fu_2631_p4 = {{ap_phi_reg_pp0_iter7_q_val_0_4165167189_reg_1057[22:7]}};

assign tmp_39_fu_2679_p4 = {{ap_phi_reg_pp0_iter7_q_val_0_5177179_reg_1102[22:7]}};

assign tmp_40_fu_2727_p4 = {{ap_phi_reg_pp0_iter7_q_val_0_6191_reg_1048[22:7]}};

assign tmp_41_fu_2775_p4 = {{ap_phi_reg_pp0_iter7_q_val_32_reg_1111[22:7]}};

assign tmp_9_reg_3483_pp0_iter4_reg = tmp_9_reg_3483;

assign tmp_s_fu_1699_p3 = {{lshr_ln_reg_3446}, {7'd0}};

assign trunc_ln10_fu_1733_p1 = x_2_fu_1726_p3[0:0];

assign trunc_ln144_fu_1158_p1 = ap_sig_allocacmp_cp_1[2:0];

assign trunc_ln154_fu_1965_p1 = ap_phi_reg_pp0_iter5_s_val_reg_953[3:0];

assign trunc_ln168_1_fu_2503_p1 = ap_phi_reg_pp0_iter7_q_val_0_1141143153159173183_reg_1084[7:0];

assign trunc_ln168_2_fu_2551_p1 = ap_phi_reg_pp0_iter7_q_val_0_2147149163169187_reg_1066[7:0];

assign trunc_ln168_3_fu_2599_p1 = ap_phi_reg_pp0_iter7_q_val_0_3155157175181_reg_1093[7:0];

assign trunc_ln168_4_fu_2647_p1 = ap_phi_reg_pp0_iter7_q_val_0_4165167189_reg_1057[7:0];

assign trunc_ln168_5_fu_2695_p1 = ap_phi_reg_pp0_iter7_q_val_0_5177179_reg_1102[7:0];

assign trunc_ln168_6_fu_2743_p1 = ap_phi_reg_pp0_iter7_q_val_0_6191_reg_1048[7:0];

assign trunc_ln168_7_fu_2791_p1 = ap_phi_reg_pp0_iter7_q_val_32_reg_1111[7:0];

assign trunc_ln168_fu_2455_p1 = ap_phi_reg_pp0_iter7_q_val_0137139145151161171185_reg_1075[7:0];

assign trunc_ln169_fu_1920_p1 = add_ln146_reg_3440_pp0_iter4_reg[2:0];

assign trunc_ln224_fu_1226_p1 = select_ln150_fu_1221_p3[21:0];

assign vq_buf_0_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_0_d1 = shl_ln169_1_reg_3811;

assign vq_buf_1_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_1_d1 = shl_ln169_3_reg_3828;

assign vq_buf_2_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_2_d1 = shl_ln169_4_reg_3833;

assign vq_buf_3_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_3_d1 = shl_ln169_5_reg_3838;

assign vq_buf_4_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_4_d1 = shl_ln169_6_reg_3843;

assign vq_buf_5_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_5_d1 = shl_ln169_7_reg_3848;

assign vq_buf_6_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_6_d1 = shl_ln169_8_reg_3853;

assign vq_buf_7_address1 = zext_ln169_reg_3458_pp0_iter7_reg;

assign vq_buf_7_d1 = shl_ln169_9_reg_3858;

assign vq_cache_o_stream_TDATA = {{{{{{{{q_out_reg_3805}, {q_out_14_reg_3799}}, {q_out_13_reg_3793}}, {q_out_12_reg_3787}}, {q_out_11_reg_3781}}, {q_out_10_reg_3775}}, {q_out_9_reg_3769}}, {q_out_8_reg_3763}};

assign vs_buf_address0 = zext_ln169_fu_1711_p1;

assign vs_buf_address1 = vs_buf_addr_reg_3472_pp0_iter5_reg;

assign vs_buf_d1 = (shl_ln172_1_reg_3629 | and_ln172_fu_2216_p2);

assign vs_cache_o_stream_TDATA = scale_reg_3591_pp0_iter6_reg;

assign x_1_fu_1721_p2 = ($signed(abs_max_8_reg_3451) + $signed(23'd8388607));

assign x_2_fu_1726_p3 = ((icmp_ln12_fu_1716_p2[0:0] == 1'b1) ? x_1_fu_1721_p2 : abs_max_8_reg_3451);

assign xor_ln154_fu_1969_p2 = (tmp_31_fu_1941_p3 ^ 1'd1);

assign xor_ln172_fu_2211_p2 = (shl_ln172_reg_3586_pp0_iter5_reg ^ 32'd4294967295);

assign zext_ln144_fu_1671_p1 = cp_1_reg_3233_pp0_iter3_reg;

assign zext_ln147_fu_1289_p1 = abs_max_reg_3271;

assign zext_ln169_10_fu_2892_p1 = q_out_reg_3805;

assign zext_ln169_1_fu_2817_p1 = shl_ln169_reg_3638_pp0_iter6_reg;

assign zext_ln169_2_fu_2820_p1 = q_out_8_reg_3763;

assign zext_ln169_3_fu_2829_p1 = trunc_ln169_reg_3576_pp0_iter6_reg;

assign zext_ln169_4_fu_2838_p1 = q_out_9_reg_3769;

assign zext_ln169_5_fu_2847_p1 = q_out_10_reg_3775;

assign zext_ln169_6_fu_2856_p1 = q_out_11_reg_3781;

assign zext_ln169_7_fu_2865_p1 = q_out_12_reg_3787;

assign zext_ln169_8_fu_2874_p1 = q_out_13_reg_3793;

assign zext_ln169_9_fu_2883_p1 = q_out_14_reg_3799;

assign zext_ln169_fu_1711_p1 = add_ln169_fu_1706_p2;

assign zext_ln172_1_fu_2029_p1 = scale_fu_1989_p3;

assign zext_ln172_fu_1931_p1 = shl_ln1_fu_1923_p3;

always @ (posedge ap_clk) begin
    mul_reg_3228[2:0] <= 3'b000;
    zext_ln169_reg_3458[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln169_reg_3458_pp0_iter5_reg[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln169_reg_3458_pp0_iter6_reg[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln169_reg_3458_pp0_iter7_reg[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln172_reg_3581[1:0] <= 2'b00;
    zext_ln172_reg_3581[31:5] <= 27'b000000000000000000000000000;
    shl_ln169_reg_3638[2:0] <= 3'b000;
    shl_ln169_reg_3638_pp0_iter6_reg[2:0] <= 3'b000;
end

endmodule //RV_GEMM_quantize_v_group
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_load_r_row (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        rq_stream_TVALID,
        rs_stream_TVALID,
        rq_stream_TDATA,
        rq_stream_TREADY,
        rs_stream_TDATA,
        rs_stream_TREADY,
        rq_buf_0_address1,
        rq_buf_0_ce1,
        rq_buf_0_we1,
        rq_buf_0_d1,
        rq_buf_1_address1,
        rq_buf_1_ce1,
        rq_buf_1_we1,
        rq_buf_1_d1,
        rq_buf_2_address1,
        rq_buf_2_ce1,
        rq_buf_2_we1,
        rq_buf_2_d1,
        rq_buf_3_address1,
        rq_buf_3_ce1,
        rq_buf_3_we1,
        rq_buf_3_d1,
        rq_buf_4_address1,
        rq_buf_4_ce1,
        rq_buf_4_we1,
        rq_buf_4_d1,
        rq_buf_5_address1,
        rq_buf_5_ce1,
        rq_buf_5_we1,
        rq_buf_5_d1,
        rq_buf_6_address1,
        rq_buf_6_ce1,
        rq_buf_6_we1,
        rq_buf_6_d1,
        rq_buf_7_address1,
        rq_buf_7_ce1,
        rq_buf_7_we1,
        rq_buf_7_d1,
        rs_buf_address1,
        rs_buf_ce1,
        rs_buf_we1,
        rs_buf_d1,
        valid_st
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
input   rq_stream_TVALID;
input   rs_stream_TVALID;
input  [63:0] rq_stream_TDATA;
output   rq_stream_TREADY;
input  [7:0] rs_stream_TDATA;
output   rs_stream_TREADY;
output  [6:0] rq_buf_0_address1;
output   rq_buf_0_ce1;
output   rq_buf_0_we1;
output  [7:0] rq_buf_0_d1;
output  [6:0] rq_buf_1_address1;
output   rq_buf_1_ce1;
output   rq_buf_1_we1;
output  [7:0] rq_buf_1_d1;
output  [6:0] rq_buf_2_address1;
output   rq_buf_2_ce1;
output   rq_buf_2_we1;
output  [7:0] rq_buf_2_d1;
output  [6:0] rq_buf_3_address1;
output   rq_buf_3_ce1;
output   rq_buf_3_we1;
output  [7:0] rq_buf_3_d1;
output  [6:0] rq_buf_4_address1;
output   rq_buf_4_ce1;
output   rq_buf_4_we1;
output  [7:0] rq_buf_4_d1;
output  [6:0] rq_buf_5_address1;
output   rq_buf_5_ce1;
output   rq_buf_5_we1;
output  [7:0] rq_buf_5_d1;
output  [6:0] rq_buf_6_address1;
output   rq_buf_6_ce1;
output   rq_buf_6_we1;
output  [7:0] rq_buf_6_d1;
output  [6:0] rq_buf_7_address1;
output   rq_buf_7_ce1;
output   rq_buf_7_we1;
output  [7:0] rq_buf_7_d1;
output  [6:0] rs_buf_address1;
output   rs_buf_ce1;
output   rs_buf_we1;
output  [3:0] rs_buf_d1;
input  [7:0] valid_st;

reg ap_idle;
reg rq_stream_TREADY;
reg rs_stream_TREADY;
reg rq_buf_0_ce1;
reg rq_buf_0_we1;
reg rq_buf_1_ce1;
reg rq_buf_1_we1;
reg rq_buf_2_ce1;
reg rq_buf_2_we1;
reg rq_buf_3_ce1;
reg rq_buf_3_we1;
reg rq_buf_4_ce1;
reg rq_buf_4_we1;
reg rq_buf_5_ce1;
reg rq_buf_5_we1;
reg rq_buf_6_ce1;
reg rq_buf_6_we1;
reg rq_buf_7_ce1;
reg rq_buf_7_we1;
reg rs_buf_ce1;
reg rs_buf_we1;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln247_fu_285_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    rq_stream_TDATA_blk_n;
reg    rs_stream_TDATA_blk_n;
reg   [7:0] st_2_reg_399;
reg   [0:0] icmp_ln247_reg_404;
wire   [0:0] icmp_ln247_reg_404_pp0_iter0_reg;
wire   [7:0] trunc_ln250_fu_297_p1;
reg   [7:0] trunc_ln250_reg_408;
reg   [7:0] trunc_ln250_1_reg_413;
reg   [7:0] trunc_ln250_2_reg_418;
reg   [7:0] trunc_ln250_3_reg_423;
reg   [7:0] trunc_ln250_4_reg_428;
reg   [7:0] trunc_ln250_5_reg_433;
reg   [7:0] trunc_ln250_6_reg_438;
reg   [7:0] trunc_ln250_7_reg_443;
wire   [3:0] trunc_ln251_fu_371_p1;
reg   [3:0] trunc_ln251_reg_448;
wire   [63:0] zext_ln247_fu_380_p1;
reg   [7:0] st_fu_102;
wire   [7:0] add_ln247_fu_291_p2;
wire    ap_loop_init;
reg   [7:0] ap_sig_allocacmp_st_2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 st_fu_102 = 8'd0;
//#0 ap_done_reg = 1'b0;
end

RV_GEMM_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if (((1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((ap_loop_exit_ready == 1'b0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= 1'b0;
    end else if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        if ((icmp_ln247_fu_285_p2 == 1'd0)) begin
            st_fu_102 <= add_ln247_fu_291_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            st_fu_102 <= 8'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln247_reg_404 <= icmp_ln247_fu_285_p2;
        st_2_reg_399 <= ap_sig_allocacmp_st_2;
        trunc_ln250_1_reg_413 <= {{rq_stream_TDATA[15:8]}};
        trunc_ln250_2_reg_418 <= {{rq_stream_TDATA[23:16]}};
        trunc_ln250_3_reg_423 <= {{rq_stream_TDATA[31:24]}};
        trunc_ln250_4_reg_428 <= {{rq_stream_TDATA[39:32]}};
        trunc_ln250_5_reg_433 <= {{rq_stream_TDATA[47:40]}};
        trunc_ln250_6_reg_438 <= {{rq_stream_TDATA[55:48]}};
        trunc_ln250_7_reg_443 <= {{rq_stream_TDATA[63:56]}};
        trunc_ln250_reg_408 <= trunc_ln250_fu_297_p1;
        trunc_ln251_reg_448 <= trunc_ln251_fu_371_p1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state2_blk = 1'b0;

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln247_fu_285_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
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
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_st_2 = 8'd0;
    end else begin
        ap_sig_allocacmp_st_2 = st_fu_102;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_0_ce1 = 1'b1;
    end else begin
        rq_buf_0_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_0_we1 = 1'b1;
    end else begin
        rq_buf_0_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_1_ce1 = 1'b1;
    end else begin
        rq_buf_1_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_1_we1 = 1'b1;
    end else begin
        rq_buf_1_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_2_ce1 = 1'b1;
    end else begin
        rq_buf_2_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_2_we1 = 1'b1;
    end else begin
        rq_buf_2_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_3_ce1 = 1'b1;
    end else begin
        rq_buf_3_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_3_we1 = 1'b1;
    end else begin
        rq_buf_3_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_4_ce1 = 1'b1;
    end else begin
        rq_buf_4_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_4_we1 = 1'b1;
    end else begin
        rq_buf_4_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_5_ce1 = 1'b1;
    end else begin
        rq_buf_5_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_5_we1 = 1'b1;
    end else begin
        rq_buf_5_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_6_ce1 = 1'b1;
    end else begin
        rq_buf_6_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_6_we1 = 1'b1;
    end else begin
        rq_buf_6_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_7_ce1 = 1'b1;
    end else begin
        rq_buf_7_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rq_buf_7_we1 = 1'b1;
    end else begin
        rq_buf_7_we1 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln247_fu_285_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        rq_stream_TDATA_blk_n = rq_stream_TVALID;
    end else begin
        rq_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln247_fu_285_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        rq_stream_TREADY = 1'b1;
    end else begin
        rq_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rs_buf_ce1 = 1'b1;
    end else begin
        rs_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd0))) begin
        rs_buf_we1 = 1'b1;
    end else begin
        rs_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln247_fu_285_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        rs_stream_TDATA_blk_n = rs_stream_TVALID;
    end else begin
        rs_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln247_fu_285_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        rs_stream_TREADY = 1'b1;
    end else begin
        rs_stream_TREADY = 1'b0;
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
            if (((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0)))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else if ((((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1)) | ((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln247_reg_404_pp0_iter0_reg == 1'd1)))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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

assign add_ln247_fu_291_p2 = (ap_sig_allocacmp_st_2 + 8'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((rs_stream_TVALID == 1'b0) & (icmp_ln247_fu_285_p2 == 1'd0)) | ((icmp_ln247_fu_285_p2 == 1'd0) & (rq_stream_TVALID == 1'b0)));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln247_fu_285_p2 = ((ap_sig_allocacmp_st_2 == valid_st) ? 1'b1 : 1'b0);

assign icmp_ln247_reg_404_pp0_iter0_reg = icmp_ln247_reg_404;

assign rq_buf_0_address1 = zext_ln247_fu_380_p1;

assign rq_buf_0_d1 = trunc_ln250_reg_408;

assign rq_buf_1_address1 = zext_ln247_fu_380_p1;

assign rq_buf_1_d1 = trunc_ln250_1_reg_413;

assign rq_buf_2_address1 = zext_ln247_fu_380_p1;

assign rq_buf_2_d1 = trunc_ln250_2_reg_418;

assign rq_buf_3_address1 = zext_ln247_fu_380_p1;

assign rq_buf_3_d1 = trunc_ln250_3_reg_423;

assign rq_buf_4_address1 = zext_ln247_fu_380_p1;

assign rq_buf_4_d1 = trunc_ln250_4_reg_428;

assign rq_buf_5_address1 = zext_ln247_fu_380_p1;

assign rq_buf_5_d1 = trunc_ln250_5_reg_433;

assign rq_buf_6_address1 = zext_ln247_fu_380_p1;

assign rq_buf_6_d1 = trunc_ln250_6_reg_438;

assign rq_buf_7_address1 = zext_ln247_fu_380_p1;

assign rq_buf_7_d1 = trunc_ln250_7_reg_443;

assign rs_buf_address1 = zext_ln247_fu_380_p1;

assign rs_buf_d1 = trunc_ln251_reg_448;

assign trunc_ln250_fu_297_p1 = rq_stream_TDATA[7:0];

assign trunc_ln251_fu_371_p1 = rs_stream_TDATA[3:0];

assign zext_ln247_fu_380_p1 = st_2_reg_399;

endmodule //RV_GEMM_load_r_row
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_quantize_a_vec (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        p_read,
        p_read1,
        p_read2,
        p_read3,
        p_read4,
        p_read5,
        p_read6,
        p_read7,
        aq_stream_TDATA,
        aq_stream_TVALID,
        aq_stream_TREADY,
        as_stream_TDATA,
        as_stream_TVALID,
        as_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 18'd1;
parameter    ap_ST_fsm_state2 = 18'd2;
parameter    ap_ST_fsm_state3 = 18'd4;
parameter    ap_ST_fsm_state4 = 18'd8;
parameter    ap_ST_fsm_state5 = 18'd16;
parameter    ap_ST_fsm_state6 = 18'd32;
parameter    ap_ST_fsm_state7 = 18'd64;
parameter    ap_ST_fsm_state8 = 18'd128;
parameter    ap_ST_fsm_state9 = 18'd256;
parameter    ap_ST_fsm_state10 = 18'd512;
parameter    ap_ST_fsm_state11 = 18'd1024;
parameter    ap_ST_fsm_state12 = 18'd2048;
parameter    ap_ST_fsm_state13 = 18'd4096;
parameter    ap_ST_fsm_state14 = 18'd8192;
parameter    ap_ST_fsm_state15 = 18'd16384;
parameter    ap_ST_fsm_state16 = 18'd32768;
parameter    ap_ST_fsm_state17 = 18'd65536;
parameter    ap_ST_fsm_state18 = 18'd131072;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [22:0] p_read;
input  [22:0] p_read1;
input  [22:0] p_read2;
input  [22:0] p_read3;
input  [22:0] p_read4;
input  [22:0] p_read5;
input  [22:0] p_read6;
input  [22:0] p_read7;
output  [63:0] aq_stream_TDATA;
output   aq_stream_TVALID;
input   aq_stream_TREADY;
output  [7:0] as_stream_TDATA;
output   as_stream_TVALID;
input   as_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg aq_stream_TVALID;
reg as_stream_TVALID;

(* fsm_encoding = "none" *) reg   [17:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    aq_stream_TDATA_blk_n;
wire    ap_CS_fsm_state17;
reg    as_stream_TDATA_blk_n;
wire   [22:0] sub_ln273_fu_179_p2;
reg   [22:0] sub_ln273_reg_1366;
reg   [0:0] tmp_reg_1371;
wire    ap_CS_fsm_state2;
wire   [21:0] select_ln273_1_fu_208_p3;
reg   [21:0] select_ln273_1_reg_1383;
wire   [22:0] select_ln273_2_fu_230_p3;
reg   [22:0] select_ln273_2_reg_1388;
wire    ap_CS_fsm_state3;
wire   [22:0] select_ln273_3_fu_246_p3;
reg   [22:0] select_ln273_3_reg_1401;
wire   [22:0] select_ln273_4_fu_267_p3;
reg   [22:0] select_ln273_4_reg_1407;
wire    ap_CS_fsm_state4;
wire   [22:0] select_ln273_5_fu_279_p3;
reg   [22:0] select_ln273_5_reg_1420;
wire   [22:0] select_ln273_6_fu_299_p3;
reg   [22:0] select_ln273_6_reg_1426;
wire    ap_CS_fsm_state5;
wire   [22:0] select_ln273_7_fu_311_p3;
reg   [22:0] select_ln273_7_reg_1439;
wire   [22:0] select_ln273_8_fu_331_p3;
reg   [22:0] select_ln273_8_reg_1445;
wire    ap_CS_fsm_state6;
wire   [22:0] select_ln273_9_fu_343_p3;
reg   [22:0] select_ln273_9_reg_1458;
wire   [22:0] select_ln273_10_fu_363_p3;
reg   [22:0] select_ln273_10_reg_1464;
wire    ap_CS_fsm_state7;
wire   [22:0] select_ln273_11_fu_375_p3;
reg   [22:0] select_ln273_11_reg_1477;
wire   [22:0] select_ln273_12_fu_395_p3;
reg   [22:0] select_ln273_12_reg_1483;
wire    ap_CS_fsm_state8;
wire   [22:0] select_ln273_13_fu_407_p3;
reg   [22:0] select_ln273_13_reg_1496;
wire   [22:0] select_ln273_14_fu_427_p3;
reg   [22:0] select_ln273_14_reg_1502;
wire   [22:0] abs_max_fu_439_p3;
reg   [22:0] abs_max_reg_1508;
wire    ap_CS_fsm_state9;
wire   [22:0] x_4_fu_455_p3;
reg   [22:0] x_4_reg_1515;
wire    ap_CS_fsm_state10;
wire   [0:0] grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_return;
reg   [0:0] targetBlock_reg_1520;
wire    ap_CS_fsm_state11;
wire   [5:0] add_ln17_fu_470_p2;
wire    ap_CS_fsm_state12;
wire   [3:0] scale_fu_524_p3;
reg   [3:0] scale_reg_1529;
wire    ap_CS_fsm_state13;
wire  signed [4:0] sub_i_i_fu_536_p2;
reg  signed [4:0] sub_i_i_reg_1535;
reg   [0:0] tmp_51_reg_1540;
wire   [4:0] conv_i_i9_i_fu_550_p2;
reg   [4:0] conv_i_i9_i_reg_1552;
wire   [22:0] shl_ln286_fu_566_p2;
reg   [22:0] shl_ln286_reg_1557;
wire    ap_CS_fsm_state14;
wire   [22:0] ashr_ln286_fu_575_p2;
reg   [22:0] ashr_ln286_reg_1562;
wire   [22:0] shl_ln286_1_fu_584_p2;
reg   [22:0] shl_ln286_1_reg_1567;
wire   [22:0] ashr_ln286_1_fu_593_p2;
reg   [22:0] ashr_ln286_1_reg_1572;
wire   [22:0] shl_ln286_2_fu_602_p2;
reg   [22:0] shl_ln286_2_reg_1577;
wire   [22:0] ashr_ln286_2_fu_611_p2;
reg   [22:0] ashr_ln286_2_reg_1582;
wire   [22:0] shl_ln286_3_fu_620_p2;
reg   [22:0] shl_ln286_3_reg_1587;
wire   [22:0] ashr_ln286_3_fu_629_p2;
reg   [22:0] ashr_ln286_3_reg_1592;
wire   [22:0] shl_ln286_4_fu_638_p2;
reg   [22:0] shl_ln286_4_reg_1597;
wire   [22:0] ashr_ln286_4_fu_647_p2;
reg   [22:0] ashr_ln286_4_reg_1602;
wire   [22:0] shl_ln286_5_fu_656_p2;
reg   [22:0] shl_ln286_5_reg_1607;
wire   [22:0] ashr_ln286_5_fu_665_p2;
reg   [22:0] ashr_ln286_5_reg_1612;
wire   [22:0] shl_ln286_6_fu_674_p2;
reg   [22:0] shl_ln286_6_reg_1617;
wire   [22:0] ashr_ln286_6_fu_683_p2;
reg   [22:0] ashr_ln286_6_reg_1622;
wire   [22:0] shl_ln286_7_fu_692_p2;
reg   [22:0] shl_ln286_7_reg_1627;
wire   [22:0] ashr_ln286_7_fu_701_p2;
reg   [22:0] ashr_ln286_7_reg_1632;
wire   [0:0] lnot_i_i_fu_706_p2;
reg   [0:0] lnot_i_i_reg_1637;
wire    ap_CS_fsm_state15;
reg   [21:0] q_val_34_reg_1649;
reg   [21:0] q_val_37_reg_1654;
reg   [21:0] q_val_40_reg_1659;
reg   [21:0] q_val_43_reg_1664;
reg   [21:0] q_val_46_reg_1669;
reg   [21:0] q_val_49_reg_1674;
reg   [21:0] q_val_52_reg_1679;
reg   [21:0] q_val_55_reg_1684;
wire   [7:0] select_ln291_1_fu_991_p3;
reg   [7:0] select_ln291_1_reg_1689;
wire    ap_CS_fsm_state16;
wire   [7:0] select_ln291_3_fu_1039_p3;
reg   [7:0] select_ln291_3_reg_1694;
wire   [7:0] select_ln291_5_fu_1087_p3;
reg   [7:0] select_ln291_5_reg_1699;
wire   [7:0] select_ln291_7_fu_1135_p3;
reg   [7:0] select_ln291_7_reg_1704;
wire   [7:0] select_ln291_9_fu_1183_p3;
reg   [7:0] select_ln291_9_reg_1709;
wire   [7:0] select_ln291_11_fu_1231_p3;
reg   [7:0] select_ln291_11_reg_1714;
wire   [7:0] select_ln291_13_fu_1279_p3;
reg   [7:0] select_ln291_13_reg_1719;
wire   [7:0] select_ln291_15_fu_1327_p3;
reg   [7:0] select_ln291_15_reg_1724;
wire    grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start;
wire    grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_done;
wire    grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_idle;
wire    grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_ready;
wire   [4:0] grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_i_out;
wire    grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_i_out_ap_vld;
reg   [5:0] s_val_reg_162;
reg    grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start_reg;
reg   [4:0] i_loc_fu_96;
reg    ap_block_state17;
reg    ap_block_state17_io;
wire   [22:0] select_ln273_fu_193_p3;
wire   [0:0] icmp_ln224_fu_202_p2;
wire   [21:0] trunc_ln224_fu_198_p1;
wire   [0:0] tmp_42_fu_222_p3;
wire   [22:0] sub_ln273_1_fu_216_p2;
wire   [22:0] zext_ln273_fu_238_p1;
wire   [0:0] icmp_ln224_8_fu_241_p2;
wire   [0:0] tmp_43_fu_259_p3;
wire   [22:0] sub_ln273_2_fu_253_p2;
wire   [0:0] icmp_ln224_9_fu_275_p2;
wire   [0:0] tmp_44_fu_291_p3;
wire   [22:0] sub_ln273_3_fu_285_p2;
wire   [0:0] icmp_ln224_10_fu_307_p2;
wire   [0:0] tmp_45_fu_323_p3;
wire   [22:0] sub_ln273_4_fu_317_p2;
wire   [0:0] icmp_ln224_11_fu_339_p2;
wire   [0:0] tmp_46_fu_355_p3;
wire   [22:0] sub_ln273_5_fu_349_p2;
wire   [0:0] icmp_ln224_12_fu_371_p2;
wire   [0:0] tmp_47_fu_387_p3;
wire   [22:0] sub_ln273_6_fu_381_p2;
wire   [0:0] icmp_ln224_13_fu_403_p2;
wire   [0:0] tmp_48_fu_419_p3;
wire   [22:0] sub_ln273_7_fu_413_p2;
wire   [0:0] icmp_ln224_14_fu_435_p2;
wire   [0:0] icmp_ln12_fu_445_p2;
wire   [22:0] x_3_fu_450_p2;
wire   [5:0] zext_ln17_fu_466_p1;
wire   [1:0] tmp_50_fu_484_p4;
wire   [0:0] tmp_49_fu_476_p3;
wire   [0:0] xor_ln277_fu_504_p2;
wire   [0:0] icmp_ln43_fu_494_p2;
wire   [0:0] or_ln277_fu_518_p2;
wire   [3:0] select_ln277_fu_510_p3;
wire   [3:0] trunc_ln277_fu_500_p1;
wire   [4:0] scale_cast_fu_532_p1;
wire  signed [31:0] conv_i_i9_i_cast_fu_556_p1;
wire   [22:0] conv_i_i9_i_castcast_fu_562_p1;
wire  signed [31:0] conv_i_i_i39_fu_559_p1;
wire   [22:0] conv_i_i_i39cast_fu_571_p1;
wire   [22:0] conv_i_i9_i_castcast94_fu_580_p1;
wire   [22:0] conv_i_i_i39cast95_fu_589_p1;
wire   [22:0] conv_i_i9_i_castcast96_fu_598_p1;
wire   [22:0] conv_i_i_i39cast97_fu_607_p1;
wire   [22:0] conv_i_i9_i_castcast98_fu_616_p1;
wire   [22:0] conv_i_i_i39cast99_fu_625_p1;
wire   [22:0] conv_i_i9_i_castcast100_fu_634_p1;
wire   [22:0] conv_i_i_i39cast101_fu_643_p1;
wire   [22:0] conv_i_i9_i_castcast102_fu_652_p1;
wire   [22:0] conv_i_i_i39cast103_fu_661_p1;
wire   [22:0] conv_i_i9_i_castcast104_fu_670_p1;
wire   [22:0] conv_i_i_i39cast105_fu_679_p1;
wire   [22:0] conv_i_i9_i_castcast106_fu_688_p1;
wire   [22:0] conv_i_i_i39cast107_fu_697_p1;
wire   [22:0] q_val_fu_711_p3;
wire   [22:0] q_val_33_fu_716_p2;
wire   [22:0] q_val_35_fu_732_p3;
wire   [22:0] q_val_36_fu_737_p2;
wire   [22:0] q_val_38_fu_753_p3;
wire   [22:0] q_val_39_fu_758_p2;
wire   [22:0] q_val_41_fu_774_p3;
wire   [22:0] q_val_42_fu_779_p2;
wire   [22:0] q_val_44_fu_795_p3;
wire   [22:0] q_val_45_fu_800_p2;
wire   [22:0] q_val_47_fu_816_p3;
wire   [22:0] q_val_48_fu_821_p2;
wire   [22:0] q_val_50_fu_837_p3;
wire   [22:0] q_val_51_fu_842_p2;
wire   [22:0] q_val_53_fu_858_p3;
wire   [22:0] q_val_54_fu_863_p2;
wire  signed [22:0] sext_ln288_6_fu_897_p1;
wire  signed [22:0] sext_ln288_4_fu_891_p1;
wire  signed [22:0] sext_ln288_2_fu_885_p1;
wire  signed [22:0] sext_ln288_fu_879_p1;
wire  signed [22:0] sext_ln288_1_fu_882_p1;
wire  signed [22:0] sext_ln288_3_fu_888_p1;
wire  signed [22:0] sext_ln288_5_fu_894_p1;
wire  signed [22:0] sext_ln288_7_fu_900_p1;
wire   [22:0] q_val_0174176182188198208222_fu_921_p3;
wire   [15:0] tmp_52_fu_957_p4;
wire   [0:0] icmp_ln42_fu_951_p2;
wire   [0:0] icmp_ln43_9_fu_967_p2;
wire   [0:0] or_ln291_fu_985_p2;
wire   [7:0] select_ln291_fu_977_p3;
wire   [7:0] trunc_ln291_fu_973_p1;
wire   [22:0] q_val_60_fu_927_p3;
wire   [15:0] tmp_53_fu_1005_p4;
wire   [0:0] icmp_ln42_8_fu_999_p2;
wire   [0:0] icmp_ln43_10_fu_1015_p2;
wire   [0:0] or_ln291_1_fu_1033_p2;
wire   [7:0] select_ln291_2_fu_1025_p3;
wire   [7:0] trunc_ln291_1_fu_1021_p1;
wire   [22:0] q_val_59_fu_915_p3;
wire   [15:0] tmp_54_fu_1053_p4;
wire   [0:0] icmp_ln42_9_fu_1047_p2;
wire   [0:0] icmp_ln43_11_fu_1063_p2;
wire   [0:0] or_ln291_2_fu_1081_p2;
wire   [7:0] select_ln291_4_fu_1073_p3;
wire   [7:0] trunc_ln291_2_fu_1069_p1;
wire   [22:0] q_val_61_fu_933_p3;
wire   [15:0] tmp_55_fu_1101_p4;
wire   [0:0] icmp_ln42_10_fu_1095_p2;
wire   [0:0] icmp_ln43_12_fu_1111_p2;
wire   [0:0] or_ln291_3_fu_1129_p2;
wire   [7:0] select_ln291_6_fu_1121_p3;
wire   [7:0] trunc_ln291_3_fu_1117_p1;
wire   [22:0] q_val_58_fu_909_p3;
wire   [15:0] tmp_56_fu_1149_p4;
wire   [0:0] icmp_ln42_11_fu_1143_p2;
wire   [0:0] icmp_ln43_13_fu_1159_p2;
wire   [0:0] or_ln291_4_fu_1177_p2;
wire   [7:0] select_ln291_8_fu_1169_p3;
wire   [7:0] trunc_ln291_4_fu_1165_p1;
wire   [22:0] q_val_62_fu_939_p3;
wire   [15:0] tmp_57_fu_1197_p4;
wire   [0:0] icmp_ln42_12_fu_1191_p2;
wire   [0:0] icmp_ln43_14_fu_1207_p2;
wire   [0:0] or_ln291_5_fu_1225_p2;
wire   [7:0] select_ln291_10_fu_1217_p3;
wire   [7:0] trunc_ln291_5_fu_1213_p1;
wire   [22:0] q_val_57_fu_903_p3;
wire   [15:0] tmp_58_fu_1245_p4;
wire   [0:0] icmp_ln42_13_fu_1239_p2;
wire   [0:0] icmp_ln43_15_fu_1255_p2;
wire   [0:0] or_ln291_6_fu_1273_p2;
wire   [7:0] select_ln291_12_fu_1265_p3;
wire   [7:0] trunc_ln291_6_fu_1261_p1;
wire   [22:0] q_val_56_fu_945_p3;
wire   [15:0] tmp_59_fu_1293_p4;
wire   [0:0] icmp_ln42_14_fu_1287_p2;
wire   [0:0] icmp_ln43_16_fu_1303_p2;
wire   [0:0] or_ln291_7_fu_1321_p2;
wire   [7:0] select_ln291_14_fu_1313_p3;
wire   [7:0] trunc_ln291_7_fu_1309_p1;
wire    ap_CS_fsm_state18;
reg   [17:0] ap_NS_fsm;
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
wire    ap_ST_fsm_state12_blk;
wire    ap_ST_fsm_state13_blk;
wire    ap_ST_fsm_state14_blk;
wire    ap_ST_fsm_state15_blk;
wire    ap_ST_fsm_state16_blk;
reg    ap_ST_fsm_state17_blk;
wire    ap_ST_fsm_state18_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 18'd1;
//#0 grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start_reg = 1'b0;
end

RV_GEMM_quantize_a_vec_Pipeline_VITIS_LOOP_15_1 grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start),
    .ap_done(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_done),
    .ap_idle(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_idle),
    .ap_ready(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_ready),
    .x_4(x_4_reg_1515),
    .i_out(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_i_out),
    .i_out_ap_vld(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_i_out_ap_vld),
    .ap_return(grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_return)
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
        grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state10)) begin
            grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start_reg <= 1'b1;
        end else if ((grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_ready == 1'b1)) begin
            grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state12)) begin
        if ((targetBlock_reg_1520 == 1'd1)) begin
            s_val_reg_162 <= 6'd57;
        end else if ((targetBlock_reg_1520 == 1'd0)) begin
            s_val_reg_162 <= add_ln17_fu_470_p2;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        abs_max_reg_1508 <= abs_max_fu_439_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        ashr_ln286_1_reg_1572 <= ashr_ln286_1_fu_593_p2;
        ashr_ln286_2_reg_1582 <= ashr_ln286_2_fu_611_p2;
        ashr_ln286_3_reg_1592 <= ashr_ln286_3_fu_629_p2;
        ashr_ln286_4_reg_1602 <= ashr_ln286_4_fu_647_p2;
        ashr_ln286_5_reg_1612 <= ashr_ln286_5_fu_665_p2;
        ashr_ln286_6_reg_1622 <= ashr_ln286_6_fu_683_p2;
        ashr_ln286_7_reg_1632 <= ashr_ln286_7_fu_701_p2;
        ashr_ln286_reg_1562 <= ashr_ln286_fu_575_p2;
        shl_ln286_1_reg_1567 <= shl_ln286_1_fu_584_p2;
        shl_ln286_2_reg_1577 <= shl_ln286_2_fu_602_p2;
        shl_ln286_3_reg_1587 <= shl_ln286_3_fu_620_p2;
        shl_ln286_4_reg_1597 <= shl_ln286_4_fu_638_p2;
        shl_ln286_5_reg_1607 <= shl_ln286_5_fu_656_p2;
        shl_ln286_6_reg_1617 <= shl_ln286_6_fu_674_p2;
        shl_ln286_7_reg_1627 <= shl_ln286_7_fu_692_p2;
        shl_ln286_reg_1557 <= shl_ln286_fu_566_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state13)) begin
        conv_i_i9_i_reg_1552 <= conv_i_i9_i_fu_550_p2;
        scale_reg_1529 <= scale_fu_524_p3;
        sub_i_i_reg_1535 <= sub_i_i_fu_536_p2;
        tmp_51_reg_1540 <= sub_i_i_fu_536_p2[32'd4];
    end
end

always @ (posedge ap_clk) begin
    if (((grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_i_out_ap_vld == 1'b1) & (1'b1 == ap_CS_fsm_state11))) begin
        i_loc_fu_96 <= grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_i_out;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state15)) begin
        lnot_i_i_reg_1637 <= lnot_i_i_fu_706_p2;
        q_val_34_reg_1649 <= {{q_val_33_fu_716_p2[22:1]}};
        q_val_37_reg_1654 <= {{q_val_36_fu_737_p2[22:1]}};
        q_val_40_reg_1659 <= {{q_val_39_fu_758_p2[22:1]}};
        q_val_43_reg_1664 <= {{q_val_42_fu_779_p2[22:1]}};
        q_val_46_reg_1669 <= {{q_val_45_fu_800_p2[22:1]}};
        q_val_49_reg_1674 <= {{q_val_48_fu_821_p2[22:1]}};
        q_val_52_reg_1679 <= {{q_val_51_fu_842_p2[22:1]}};
        q_val_55_reg_1684 <= {{q_val_54_fu_863_p2[22:1]}};
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        select_ln273_10_reg_1464 <= select_ln273_10_fu_363_p3;
        select_ln273_9_reg_1458 <= select_ln273_9_fu_343_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        select_ln273_11_reg_1477 <= select_ln273_11_fu_375_p3;
        select_ln273_12_reg_1483 <= select_ln273_12_fu_395_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        select_ln273_13_reg_1496 <= select_ln273_13_fu_407_p3;
        select_ln273_14_reg_1502 <= select_ln273_14_fu_427_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        select_ln273_1_reg_1383 <= select_ln273_1_fu_208_p3;
        select_ln273_2_reg_1388 <= select_ln273_2_fu_230_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        select_ln273_3_reg_1401 <= select_ln273_3_fu_246_p3;
        select_ln273_4_reg_1407 <= select_ln273_4_fu_267_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        select_ln273_5_reg_1420 <= select_ln273_5_fu_279_p3;
        select_ln273_6_reg_1426 <= select_ln273_6_fu_299_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        select_ln273_7_reg_1439 <= select_ln273_7_fu_311_p3;
        select_ln273_8_reg_1445 <= select_ln273_8_fu_331_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state16)) begin
        select_ln291_11_reg_1714 <= select_ln291_11_fu_1231_p3;
        select_ln291_13_reg_1719 <= select_ln291_13_fu_1279_p3;
        select_ln291_15_reg_1724 <= select_ln291_15_fu_1327_p3;
        select_ln291_1_reg_1689 <= select_ln291_1_fu_991_p3;
        select_ln291_3_reg_1694 <= select_ln291_3_fu_1039_p3;
        select_ln291_5_reg_1699 <= select_ln291_5_fu_1087_p3;
        select_ln291_7_reg_1704 <= select_ln291_7_fu_1135_p3;
        select_ln291_9_reg_1709 <= select_ln291_9_fu_1183_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        sub_ln273_reg_1366 <= sub_ln273_fu_179_p2;
        tmp_reg_1371 <= p_read[32'd22];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        targetBlock_reg_1520 <= grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_return;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state10)) begin
        x_4_reg_1515 <= x_4_fu_455_p3;
    end
end

assign ap_ST_fsm_state10_blk = 1'b0;

always @ (*) begin
    if ((grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_done == 1'b0)) begin
        ap_ST_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_fsm_state11_blk = 1'b0;
    end
end

assign ap_ST_fsm_state12_blk = 1'b0;

assign ap_ST_fsm_state13_blk = 1'b0;

assign ap_ST_fsm_state14_blk = 1'b0;

assign ap_ST_fsm_state15_blk = 1'b0;

assign ap_ST_fsm_state16_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_block_state17_io) | (1'b1 == ap_block_state17))) begin
        ap_ST_fsm_state17_blk = 1'b1;
    end else begin
        ap_ST_fsm_state17_blk = 1'b0;
    end
end

assign ap_ST_fsm_state18_blk = 1'b0;

always @ (*) begin
    if ((ap_start == 1'b0)) begin
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
    if (((1'b1 == ap_CS_fsm_state18) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
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
    if ((1'b1 == ap_CS_fsm_state18)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state17)) begin
        aq_stream_TDATA_blk_n = aq_stream_TREADY;
    end else begin
        aq_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state17_io) | (1'b1 == ap_block_state17)) & (1'b1 == ap_CS_fsm_state17))) begin
        aq_stream_TVALID = 1'b1;
    end else begin
        aq_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state17)) begin
        as_stream_TDATA_blk_n = as_stream_TREADY;
    end else begin
        as_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state17_io) | (1'b1 == ap_block_state17)) & (1'b1 == ap_CS_fsm_state17))) begin
        as_stream_TVALID = 1'b1;
    end else begin
        as_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
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
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state11;
        end
        ap_ST_fsm_state11 : begin
            if (((grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state11))) begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            ap_NS_fsm = ap_ST_fsm_state14;
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state15;
        end
        ap_ST_fsm_state15 : begin
            ap_NS_fsm = ap_ST_fsm_state16;
        end
        ap_ST_fsm_state16 : begin
            ap_NS_fsm = ap_ST_fsm_state17;
        end
        ap_ST_fsm_state17 : begin
            if ((~((1'b1 == ap_block_state17_io) | (1'b1 == ap_block_state17)) & (1'b1 == ap_CS_fsm_state17))) begin
                ap_NS_fsm = ap_ST_fsm_state18;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state17;
            end
        end
        ap_ST_fsm_state18 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign abs_max_fu_439_p3 = ((icmp_ln224_14_fu_435_p2[0:0] == 1'b1) ? select_ln273_14_reg_1502 : select_ln273_13_reg_1496);

assign add_ln17_fu_470_p2 = ($signed(zext_ln17_fu_466_p1) + $signed(6'd57));

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

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

always @ (*) begin
    ap_block_state17 = ((as_stream_TREADY == 1'b0) | (aq_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state17_io = ((as_stream_TREADY == 1'b0) | (aq_stream_TREADY == 1'b0));
end

assign aq_stream_TDATA = {{{{{{{{select_ln291_15_reg_1724}, {select_ln291_13_reg_1719}}, {select_ln291_11_reg_1714}}, {select_ln291_9_reg_1709}}, {select_ln291_7_reg_1704}}, {select_ln291_5_reg_1699}}, {select_ln291_3_reg_1694}}, {select_ln291_1_reg_1689}};

assign as_stream_TDATA = scale_reg_1529;

assign ashr_ln286_1_fu_593_p2 = $signed(p_read1) >>> conv_i_i_i39cast95_fu_589_p1;

assign ashr_ln286_2_fu_611_p2 = $signed(p_read2) >>> conv_i_i_i39cast97_fu_607_p1;

assign ashr_ln286_3_fu_629_p2 = $signed(p_read3) >>> conv_i_i_i39cast99_fu_625_p1;

assign ashr_ln286_4_fu_647_p2 = $signed(p_read4) >>> conv_i_i_i39cast101_fu_643_p1;

assign ashr_ln286_5_fu_665_p2 = $signed(p_read5) >>> conv_i_i_i39cast103_fu_661_p1;

assign ashr_ln286_6_fu_683_p2 = $signed(p_read6) >>> conv_i_i_i39cast105_fu_679_p1;

assign ashr_ln286_7_fu_701_p2 = $signed(p_read7) >>> conv_i_i_i39cast107_fu_697_p1;

assign ashr_ln286_fu_575_p2 = $signed(p_read) >>> conv_i_i_i39cast_fu_571_p1;

assign conv_i_i9_i_cast_fu_556_p1 = $signed(conv_i_i9_i_reg_1552);

assign conv_i_i9_i_castcast100_fu_634_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast102_fu_652_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast104_fu_670_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast106_fu_688_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast94_fu_580_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast96_fu_598_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast98_fu_616_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_castcast_fu_562_p1 = conv_i_i9_i_cast_fu_556_p1[22:0];

assign conv_i_i9_i_fu_550_p2 = (5'd1 - scale_cast_fu_532_p1);

assign conv_i_i_i39_fu_559_p1 = sub_i_i_reg_1535;

assign conv_i_i_i39cast101_fu_643_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast103_fu_661_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast105_fu_679_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast107_fu_697_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast95_fu_589_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast97_fu_607_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast99_fu_625_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign conv_i_i_i39cast_fu_571_p1 = conv_i_i_i39_fu_559_p1[22:0];

assign grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start = grp_quantize_a_vec_Pipeline_VITIS_LOOP_15_1_fu_173_ap_start_reg;

assign icmp_ln12_fu_445_p2 = (($signed(abs_max_reg_1508) > $signed(23'd0)) ? 1'b1 : 1'b0);

assign icmp_ln224_10_fu_307_p2 = (($signed(select_ln273_5_reg_1420) < $signed(select_ln273_6_reg_1426)) ? 1'b1 : 1'b0);

assign icmp_ln224_11_fu_339_p2 = (($signed(select_ln273_7_reg_1439) < $signed(select_ln273_8_reg_1445)) ? 1'b1 : 1'b0);

assign icmp_ln224_12_fu_371_p2 = (($signed(select_ln273_9_reg_1458) < $signed(select_ln273_10_reg_1464)) ? 1'b1 : 1'b0);

assign icmp_ln224_13_fu_403_p2 = (($signed(select_ln273_11_reg_1477) < $signed(select_ln273_12_reg_1483)) ? 1'b1 : 1'b0);

assign icmp_ln224_14_fu_435_p2 = (($signed(select_ln273_13_reg_1496) < $signed(select_ln273_14_reg_1502)) ? 1'b1 : 1'b0);

assign icmp_ln224_8_fu_241_p2 = (($signed(zext_ln273_fu_238_p1) < $signed(select_ln273_2_reg_1388)) ? 1'b1 : 1'b0);

assign icmp_ln224_9_fu_275_p2 = (($signed(select_ln273_3_reg_1401) < $signed(select_ln273_4_reg_1407)) ? 1'b1 : 1'b0);

assign icmp_ln224_fu_202_p2 = (($signed(select_ln273_fu_193_p3) > $signed(23'd0)) ? 1'b1 : 1'b0);

assign icmp_ln42_10_fu_1095_p2 = (($signed(q_val_61_fu_933_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_11_fu_1143_p2 = (($signed(q_val_58_fu_909_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_12_fu_1191_p2 = (($signed(q_val_62_fu_939_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_13_fu_1239_p2 = (($signed(q_val_57_fu_903_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_14_fu_1287_p2 = (($signed(q_val_56_fu_945_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_8_fu_999_p2 = (($signed(q_val_60_fu_927_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_9_fu_1047_p2 = (($signed(q_val_59_fu_915_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln42_fu_951_p2 = (($signed(q_val_0174176182188198208222_fu_921_p3) < $signed(23'd8388480)) ? 1'b1 : 1'b0);

assign icmp_ln43_10_fu_1015_p2 = (($signed(tmp_53_fu_1005_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_11_fu_1063_p2 = (($signed(tmp_54_fu_1053_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_12_fu_1111_p2 = (($signed(tmp_55_fu_1101_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_13_fu_1159_p2 = (($signed(tmp_56_fu_1149_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_14_fu_1207_p2 = (($signed(tmp_57_fu_1197_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_15_fu_1255_p2 = (($signed(tmp_58_fu_1245_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_16_fu_1303_p2 = (($signed(tmp_59_fu_1293_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_9_fu_967_p2 = (($signed(tmp_52_fu_957_p4) > $signed(16'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_fu_494_p2 = ((tmp_50_fu_484_p4 == 2'd1) ? 1'b1 : 1'b0);

assign lnot_i_i_fu_706_p2 = ((scale_reg_1529 == 4'd0) ? 1'b1 : 1'b0);

assign or_ln277_fu_518_p2 = (tmp_49_fu_476_p3 | icmp_ln43_fu_494_p2);

assign or_ln291_1_fu_1033_p2 = (icmp_ln43_10_fu_1015_p2 | icmp_ln42_8_fu_999_p2);

assign or_ln291_2_fu_1081_p2 = (icmp_ln43_11_fu_1063_p2 | icmp_ln42_9_fu_1047_p2);

assign or_ln291_3_fu_1129_p2 = (icmp_ln43_12_fu_1111_p2 | icmp_ln42_10_fu_1095_p2);

assign or_ln291_4_fu_1177_p2 = (icmp_ln43_13_fu_1159_p2 | icmp_ln42_11_fu_1143_p2);

assign or_ln291_5_fu_1225_p2 = (icmp_ln43_14_fu_1207_p2 | icmp_ln42_12_fu_1191_p2);

assign or_ln291_6_fu_1273_p2 = (icmp_ln43_15_fu_1255_p2 | icmp_ln42_13_fu_1239_p2);

assign or_ln291_7_fu_1321_p2 = (icmp_ln43_16_fu_1303_p2 | icmp_ln42_14_fu_1287_p2);

assign or_ln291_fu_985_p2 = (icmp_ln43_9_fu_967_p2 | icmp_ln42_fu_951_p2);

assign q_val_0174176182188198208222_fu_921_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read : sext_ln288_fu_879_p1);

assign q_val_33_fu_716_p2 = (q_val_fu_711_p3 + 23'd1);

assign q_val_35_fu_732_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_1_reg_1567 : ashr_ln286_1_reg_1572);

assign q_val_36_fu_737_p2 = (q_val_35_fu_732_p3 + 23'd1);

assign q_val_38_fu_753_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_2_reg_1577 : ashr_ln286_2_reg_1582);

assign q_val_39_fu_758_p2 = (q_val_38_fu_753_p3 + 23'd1);

assign q_val_41_fu_774_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_3_reg_1587 : ashr_ln286_3_reg_1592);

assign q_val_42_fu_779_p2 = (q_val_41_fu_774_p3 + 23'd1);

assign q_val_44_fu_795_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_4_reg_1597 : ashr_ln286_4_reg_1602);

assign q_val_45_fu_800_p2 = (q_val_44_fu_795_p3 + 23'd1);

assign q_val_47_fu_816_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_5_reg_1607 : ashr_ln286_5_reg_1612);

assign q_val_48_fu_821_p2 = (q_val_47_fu_816_p3 + 23'd1);

assign q_val_50_fu_837_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_6_reg_1617 : ashr_ln286_6_reg_1622);

assign q_val_51_fu_842_p2 = (q_val_50_fu_837_p3 + 23'd1);

assign q_val_53_fu_858_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_7_reg_1627 : ashr_ln286_7_reg_1632);

assign q_val_54_fu_863_p2 = (q_val_53_fu_858_p3 + 23'd1);

assign q_val_56_fu_945_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read7 : sext_ln288_7_fu_900_p1);

assign q_val_57_fu_903_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read6 : sext_ln288_6_fu_897_p1);

assign q_val_58_fu_909_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read4 : sext_ln288_4_fu_891_p1);

assign q_val_59_fu_915_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read2 : sext_ln288_2_fu_885_p1);

assign q_val_60_fu_927_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read1 : sext_ln288_1_fu_882_p1);

assign q_val_61_fu_933_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read3 : sext_ln288_3_fu_888_p1);

assign q_val_62_fu_939_p3 = ((lnot_i_i_reg_1637[0:0] == 1'b1) ? p_read5 : sext_ln288_5_fu_894_p1);

assign q_val_fu_711_p3 = ((tmp_51_reg_1540[0:0] == 1'b1) ? shl_ln286_reg_1557 : ashr_ln286_reg_1562);

assign scale_cast_fu_532_p1 = scale_fu_524_p3;

assign scale_fu_524_p3 = ((or_ln277_fu_518_p2[0:0] == 1'b1) ? select_ln277_fu_510_p3 : trunc_ln277_fu_500_p1);

assign select_ln273_10_fu_363_p3 = ((tmp_46_fu_355_p3[0:0] == 1'b1) ? sub_ln273_5_fu_349_p2 : p_read5);

assign select_ln273_11_fu_375_p3 = ((icmp_ln224_12_fu_371_p2[0:0] == 1'b1) ? select_ln273_10_reg_1464 : select_ln273_9_reg_1458);

assign select_ln273_12_fu_395_p3 = ((tmp_47_fu_387_p3[0:0] == 1'b1) ? sub_ln273_6_fu_381_p2 : p_read6);

assign select_ln273_13_fu_407_p3 = ((icmp_ln224_13_fu_403_p2[0:0] == 1'b1) ? select_ln273_12_reg_1483 : select_ln273_11_reg_1477);

assign select_ln273_14_fu_427_p3 = ((tmp_48_fu_419_p3[0:0] == 1'b1) ? sub_ln273_7_fu_413_p2 : p_read7);

assign select_ln273_1_fu_208_p3 = ((icmp_ln224_fu_202_p2[0:0] == 1'b1) ? trunc_ln224_fu_198_p1 : 22'd0);

assign select_ln273_2_fu_230_p3 = ((tmp_42_fu_222_p3[0:0] == 1'b1) ? sub_ln273_1_fu_216_p2 : p_read1);

assign select_ln273_3_fu_246_p3 = ((icmp_ln224_8_fu_241_p2[0:0] == 1'b1) ? select_ln273_2_reg_1388 : zext_ln273_fu_238_p1);

assign select_ln273_4_fu_267_p3 = ((tmp_43_fu_259_p3[0:0] == 1'b1) ? sub_ln273_2_fu_253_p2 : p_read2);

assign select_ln273_5_fu_279_p3 = ((icmp_ln224_9_fu_275_p2[0:0] == 1'b1) ? select_ln273_4_reg_1407 : select_ln273_3_reg_1401);

assign select_ln273_6_fu_299_p3 = ((tmp_44_fu_291_p3[0:0] == 1'b1) ? sub_ln273_3_fu_285_p2 : p_read3);

assign select_ln273_7_fu_311_p3 = ((icmp_ln224_10_fu_307_p2[0:0] == 1'b1) ? select_ln273_6_reg_1426 : select_ln273_5_reg_1420);

assign select_ln273_8_fu_331_p3 = ((tmp_45_fu_323_p3[0:0] == 1'b1) ? sub_ln273_4_fu_317_p2 : p_read4);

assign select_ln273_9_fu_343_p3 = ((icmp_ln224_11_fu_339_p2[0:0] == 1'b1) ? select_ln273_8_reg_1445 : select_ln273_7_reg_1439);

assign select_ln273_fu_193_p3 = ((tmp_reg_1371[0:0] == 1'b1) ? sub_ln273_reg_1366 : p_read);

assign select_ln277_fu_510_p3 = ((xor_ln277_fu_504_p2[0:0] == 1'b1) ? 4'd15 : 4'd0);

assign select_ln291_10_fu_1217_p3 = ((icmp_ln42_12_fu_1191_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_11_fu_1231_p3 = ((or_ln291_5_fu_1225_p2[0:0] == 1'b1) ? select_ln291_10_fu_1217_p3 : trunc_ln291_5_fu_1213_p1);

assign select_ln291_12_fu_1265_p3 = ((icmp_ln42_13_fu_1239_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_13_fu_1279_p3 = ((or_ln291_6_fu_1273_p2[0:0] == 1'b1) ? select_ln291_12_fu_1265_p3 : trunc_ln291_6_fu_1261_p1);

assign select_ln291_14_fu_1313_p3 = ((icmp_ln42_14_fu_1287_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_15_fu_1327_p3 = ((or_ln291_7_fu_1321_p2[0:0] == 1'b1) ? select_ln291_14_fu_1313_p3 : trunc_ln291_7_fu_1309_p1);

assign select_ln291_1_fu_991_p3 = ((or_ln291_fu_985_p2[0:0] == 1'b1) ? select_ln291_fu_977_p3 : trunc_ln291_fu_973_p1);

assign select_ln291_2_fu_1025_p3 = ((icmp_ln42_8_fu_999_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_3_fu_1039_p3 = ((or_ln291_1_fu_1033_p2[0:0] == 1'b1) ? select_ln291_2_fu_1025_p3 : trunc_ln291_1_fu_1021_p1);

assign select_ln291_4_fu_1073_p3 = ((icmp_ln42_9_fu_1047_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_5_fu_1087_p3 = ((or_ln291_2_fu_1081_p2[0:0] == 1'b1) ? select_ln291_4_fu_1073_p3 : trunc_ln291_2_fu_1069_p1);

assign select_ln291_6_fu_1121_p3 = ((icmp_ln42_10_fu_1095_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_7_fu_1135_p3 = ((or_ln291_3_fu_1129_p2[0:0] == 1'b1) ? select_ln291_6_fu_1121_p3 : trunc_ln291_3_fu_1117_p1);

assign select_ln291_8_fu_1169_p3 = ((icmp_ln42_11_fu_1143_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln291_9_fu_1183_p3 = ((or_ln291_4_fu_1177_p2[0:0] == 1'b1) ? select_ln291_8_fu_1169_p3 : trunc_ln291_4_fu_1165_p1);

assign select_ln291_fu_977_p3 = ((icmp_ln42_fu_951_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign sext_ln288_1_fu_882_p1 = $signed(q_val_37_reg_1654);

assign sext_ln288_2_fu_885_p1 = $signed(q_val_40_reg_1659);

assign sext_ln288_3_fu_888_p1 = $signed(q_val_43_reg_1664);

assign sext_ln288_4_fu_891_p1 = $signed(q_val_46_reg_1669);

assign sext_ln288_5_fu_894_p1 = $signed(q_val_49_reg_1674);

assign sext_ln288_6_fu_897_p1 = $signed(q_val_52_reg_1679);

assign sext_ln288_7_fu_900_p1 = $signed(q_val_55_reg_1684);

assign sext_ln288_fu_879_p1 = $signed(q_val_34_reg_1649);

assign shl_ln286_1_fu_584_p2 = p_read1 << conv_i_i9_i_castcast94_fu_580_p1;

assign shl_ln286_2_fu_602_p2 = p_read2 << conv_i_i9_i_castcast96_fu_598_p1;

assign shl_ln286_3_fu_620_p2 = p_read3 << conv_i_i9_i_castcast98_fu_616_p1;

assign shl_ln286_4_fu_638_p2 = p_read4 << conv_i_i9_i_castcast100_fu_634_p1;

assign shl_ln286_5_fu_656_p2 = p_read5 << conv_i_i9_i_castcast102_fu_652_p1;

assign shl_ln286_6_fu_674_p2 = p_read6 << conv_i_i9_i_castcast104_fu_670_p1;

assign shl_ln286_7_fu_692_p2 = p_read7 << conv_i_i9_i_castcast106_fu_688_p1;

assign shl_ln286_fu_566_p2 = p_read << conv_i_i9_i_castcast_fu_562_p1;

assign sub_i_i_fu_536_p2 = ($signed(scale_cast_fu_532_p1) + $signed(5'd31));

assign sub_ln273_1_fu_216_p2 = (23'd0 - p_read1);

assign sub_ln273_2_fu_253_p2 = (23'd0 - p_read2);

assign sub_ln273_3_fu_285_p2 = (23'd0 - p_read3);

assign sub_ln273_4_fu_317_p2 = (23'd0 - p_read4);

assign sub_ln273_5_fu_349_p2 = (23'd0 - p_read5);

assign sub_ln273_6_fu_381_p2 = (23'd0 - p_read6);

assign sub_ln273_7_fu_413_p2 = (23'd0 - p_read7);

assign sub_ln273_fu_179_p2 = (23'd0 - p_read);

assign tmp_42_fu_222_p3 = p_read1[32'd22];

assign tmp_43_fu_259_p3 = p_read2[32'd22];

assign tmp_44_fu_291_p3 = p_read3[32'd22];

assign tmp_45_fu_323_p3 = p_read4[32'd22];

assign tmp_46_fu_355_p3 = p_read5[32'd22];

assign tmp_47_fu_387_p3 = p_read6[32'd22];

assign tmp_48_fu_419_p3 = p_read7[32'd22];

assign tmp_49_fu_476_p3 = s_val_reg_162[32'd5];

assign tmp_50_fu_484_p4 = {{s_val_reg_162[5:4]}};

assign tmp_52_fu_957_p4 = {{q_val_0174176182188198208222_fu_921_p3[22:7]}};

assign tmp_53_fu_1005_p4 = {{q_val_60_fu_927_p3[22:7]}};

assign tmp_54_fu_1053_p4 = {{q_val_59_fu_915_p3[22:7]}};

assign tmp_55_fu_1101_p4 = {{q_val_61_fu_933_p3[22:7]}};

assign tmp_56_fu_1149_p4 = {{q_val_58_fu_909_p3[22:7]}};

assign tmp_57_fu_1197_p4 = {{q_val_62_fu_939_p3[22:7]}};

assign tmp_58_fu_1245_p4 = {{q_val_57_fu_903_p3[22:7]}};

assign tmp_59_fu_1293_p4 = {{q_val_56_fu_945_p3[22:7]}};

assign trunc_ln224_fu_198_p1 = select_ln273_fu_193_p3[21:0];

assign trunc_ln277_fu_500_p1 = s_val_reg_162[3:0];

assign trunc_ln291_1_fu_1021_p1 = q_val_60_fu_927_p3[7:0];

assign trunc_ln291_2_fu_1069_p1 = q_val_59_fu_915_p3[7:0];

assign trunc_ln291_3_fu_1117_p1 = q_val_61_fu_933_p3[7:0];

assign trunc_ln291_4_fu_1165_p1 = q_val_58_fu_909_p3[7:0];

assign trunc_ln291_5_fu_1213_p1 = q_val_62_fu_939_p3[7:0];

assign trunc_ln291_6_fu_1261_p1 = q_val_57_fu_903_p3[7:0];

assign trunc_ln291_7_fu_1309_p1 = q_val_56_fu_945_p3[7:0];

assign trunc_ln291_fu_973_p1 = q_val_0174176182188198208222_fu_921_p3[7:0];

assign x_3_fu_450_p2 = ($signed(abs_max_reg_1508) + $signed(23'd8388607));

assign x_4_fu_455_p3 = ((icmp_ln12_fu_445_p2[0:0] == 1'b1) ? x_3_fu_450_p2 : abs_max_reg_1508);

assign xor_ln277_fu_504_p2 = (tmp_49_fu_476_p3 ^ 1'd1);

assign zext_ln17_fu_466_p1 = i_loc_fu_96;

assign zext_ln273_fu_238_p1 = select_ln273_1_reg_1383;

endmodule //RV_GEMM_quantize_a_vec
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module RV_GEMM_flow_control_loop_pipe_sequential_init(
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

// if no ap_continue port and current module is not RV_GEMM module, 
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

// if no ap_continue port and current module is not RV_GEMM module, ap_done handshakes with ap_start
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

module RV_GEMM_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        v_stream_TVALID,
        v_stream_TDATA,
        v_stream_TREADY,
        p_0_0_7_0_0_0127_out,
        p_0_0_7_0_0_0127_out_ap_vld,
        p_0_0_6_0_0_0125_out,
        p_0_0_6_0_0_0125_out_ap_vld,
        p_0_0_5_0_0_0123_out,
        p_0_0_5_0_0_0123_out_ap_vld,
        p_0_0_4_0_0_0121_out,
        p_0_0_4_0_0_0121_out_ap_vld,
        p_0_0_3_0_0_0119_out,
        p_0_0_3_0_0_0119_out_ap_vld,
        p_0_0_2_0_0_0117_out,
        p_0_0_2_0_0_0117_out_ap_vld,
        p_0_0_1_0_0_0115_out,
        p_0_0_1_0_0_0115_out_ap_vld,
        p_0_0_0_0_0_0113_out,
        p_0_0_0_0_0_0113_out_ap_vld,
        p_0_0_7_0_0_0111_out,
        p_0_0_7_0_0_0111_out_ap_vld,
        p_0_0_6_0_0_0109_out,
        p_0_0_6_0_0_0109_out_ap_vld,
        p_0_0_5_0_0_0107_out,
        p_0_0_5_0_0_0107_out_ap_vld,
        p_0_0_4_0_0_0105_out,
        p_0_0_4_0_0_0105_out_ap_vld,
        p_0_0_3_0_0_0103_out,
        p_0_0_3_0_0_0103_out_ap_vld,
        p_0_0_2_0_0_0101_out,
        p_0_0_2_0_0_0101_out_ap_vld,
        p_0_0_1_0_0_099_out,
        p_0_0_1_0_0_099_out_ap_vld,
        p_0_0_0_0_0_097_out,
        p_0_0_0_0_0_097_out_ap_vld,
        p_0_0_7_0_0_095_out,
        p_0_0_7_0_0_095_out_ap_vld,
        p_0_0_6_0_0_093_out,
        p_0_0_6_0_0_093_out_ap_vld,
        p_0_0_5_0_0_091_out,
        p_0_0_5_0_0_091_out_ap_vld,
        p_0_0_4_0_0_089_out,
        p_0_0_4_0_0_089_out_ap_vld,
        p_0_0_3_0_0_087_out,
        p_0_0_3_0_0_087_out_ap_vld,
        p_0_0_2_0_0_085_out,
        p_0_0_2_0_0_085_out_ap_vld,
        p_0_0_1_0_0_083_out,
        p_0_0_1_0_0_083_out_ap_vld,
        p_0_0_0_0_0_081_out,
        p_0_0_0_0_0_081_out_ap_vld,
        p_0_0_7_0_0_079_out,
        p_0_0_7_0_0_079_out_ap_vld,
        p_0_0_6_0_0_077_out,
        p_0_0_6_0_0_077_out_ap_vld,
        p_0_0_5_0_0_075_out,
        p_0_0_5_0_0_075_out_ap_vld,
        p_0_0_4_0_0_073_out,
        p_0_0_4_0_0_073_out_ap_vld,
        p_0_0_3_0_0_071_out,
        p_0_0_3_0_0_071_out_ap_vld,
        p_0_0_2_0_0_069_out,
        p_0_0_2_0_0_069_out_ap_vld,
        p_0_0_1_0_0_067_out,
        p_0_0_1_0_0_067_out_ap_vld,
        p_0_0_0_0_0_065_out,
        p_0_0_0_0_0_065_out_ap_vld,
        p_0_0_7_0_0_063_out,
        p_0_0_7_0_0_063_out_ap_vld,
        p_0_0_6_0_0_061_out,
        p_0_0_6_0_0_061_out_ap_vld,
        p_0_0_5_0_0_059_out,
        p_0_0_5_0_0_059_out_ap_vld,
        p_0_0_4_0_0_057_out,
        p_0_0_4_0_0_057_out_ap_vld,
        p_0_0_3_0_0_055_out,
        p_0_0_3_0_0_055_out_ap_vld,
        p_0_0_2_0_0_053_out,
        p_0_0_2_0_0_053_out_ap_vld,
        p_0_0_1_0_0_051_out,
        p_0_0_1_0_0_051_out_ap_vld,
        p_0_0_0_0_0_049_out,
        p_0_0_0_0_0_049_out_ap_vld,
        p_0_0_7_0_0_047_out,
        p_0_0_7_0_0_047_out_ap_vld,
        p_0_0_6_0_0_045_out,
        p_0_0_6_0_0_045_out_ap_vld,
        p_0_0_5_0_0_043_out,
        p_0_0_5_0_0_043_out_ap_vld,
        p_0_0_4_0_0_041_out,
        p_0_0_4_0_0_041_out_ap_vld,
        p_0_0_3_0_0_039_out,
        p_0_0_3_0_0_039_out_ap_vld,
        p_0_0_2_0_0_037_out,
        p_0_0_2_0_0_037_out_ap_vld,
        p_0_0_1_0_0_035_out,
        p_0_0_1_0_0_035_out_ap_vld,
        p_0_0_0_0_0_033_out,
        p_0_0_0_0_0_033_out_ap_vld,
        p_0_0_7_0_0_031_out,
        p_0_0_7_0_0_031_out_ap_vld,
        p_0_0_6_0_0_029_out,
        p_0_0_6_0_0_029_out_ap_vld,
        p_0_0_5_0_0_027_out,
        p_0_0_5_0_0_027_out_ap_vld,
        p_0_0_4_0_0_025_out,
        p_0_0_4_0_0_025_out_ap_vld,
        p_0_0_3_0_0_023_out,
        p_0_0_3_0_0_023_out_ap_vld,
        p_0_0_2_0_0_021_out,
        p_0_0_2_0_0_021_out_ap_vld,
        p_0_0_1_0_0_019_out,
        p_0_0_1_0_0_019_out_ap_vld,
        p_0_0_0_0_0_017_out,
        p_0_0_0_0_0_017_out_ap_vld,
        p_0_0_7_0_0_015_out,
        p_0_0_7_0_0_015_out_ap_vld,
        p_0_0_6_0_0_013_out,
        p_0_0_6_0_0_013_out_ap_vld,
        p_0_0_5_0_0_011_out,
        p_0_0_5_0_0_011_out_ap_vld,
        p_0_0_4_0_0_09_out,
        p_0_0_4_0_0_09_out_ap_vld,
        p_0_0_3_0_0_07_out,
        p_0_0_3_0_0_07_out_ap_vld,
        p_0_0_2_0_0_05_out,
        p_0_0_2_0_0_05_out_ap_vld,
        p_0_0_1_0_0_03_out,
        p_0_0_1_0_0_03_out_ap_vld,
        p_0_0_0_0_0_01_out,
        p_0_0_0_0_0_01_out_ap_vld
);

parameter    ap_ST_fsm_state1 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   v_stream_TVALID;
input  [183:0] v_stream_TDATA;
output   v_stream_TREADY;
output  [22:0] p_0_0_7_0_0_0127_out;
output   p_0_0_7_0_0_0127_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0125_out;
output   p_0_0_6_0_0_0125_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0123_out;
output   p_0_0_5_0_0_0123_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0121_out;
output   p_0_0_4_0_0_0121_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0119_out;
output   p_0_0_3_0_0_0119_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0117_out;
output   p_0_0_2_0_0_0117_out_ap_vld;
output  [22:0] p_0_0_1_0_0_0115_out;
output   p_0_0_1_0_0_0115_out_ap_vld;
output  [22:0] p_0_0_0_0_0_0113_out;
output   p_0_0_0_0_0_0113_out_ap_vld;
output  [22:0] p_0_0_7_0_0_0111_out;
output   p_0_0_7_0_0_0111_out_ap_vld;
output  [22:0] p_0_0_6_0_0_0109_out;
output   p_0_0_6_0_0_0109_out_ap_vld;
output  [22:0] p_0_0_5_0_0_0107_out;
output   p_0_0_5_0_0_0107_out_ap_vld;
output  [22:0] p_0_0_4_0_0_0105_out;
output   p_0_0_4_0_0_0105_out_ap_vld;
output  [22:0] p_0_0_3_0_0_0103_out;
output   p_0_0_3_0_0_0103_out_ap_vld;
output  [22:0] p_0_0_2_0_0_0101_out;
output   p_0_0_2_0_0_0101_out_ap_vld;
output  [22:0] p_0_0_1_0_0_099_out;
output   p_0_0_1_0_0_099_out_ap_vld;
output  [22:0] p_0_0_0_0_0_097_out;
output   p_0_0_0_0_0_097_out_ap_vld;
output  [22:0] p_0_0_7_0_0_095_out;
output   p_0_0_7_0_0_095_out_ap_vld;
output  [22:0] p_0_0_6_0_0_093_out;
output   p_0_0_6_0_0_093_out_ap_vld;
output  [22:0] p_0_0_5_0_0_091_out;
output   p_0_0_5_0_0_091_out_ap_vld;
output  [22:0] p_0_0_4_0_0_089_out;
output   p_0_0_4_0_0_089_out_ap_vld;
output  [22:0] p_0_0_3_0_0_087_out;
output   p_0_0_3_0_0_087_out_ap_vld;
output  [22:0] p_0_0_2_0_0_085_out;
output   p_0_0_2_0_0_085_out_ap_vld;
output  [22:0] p_0_0_1_0_0_083_out;
output   p_0_0_1_0_0_083_out_ap_vld;
output  [22:0] p_0_0_0_0_0_081_out;
output   p_0_0_0_0_0_081_out_ap_vld;
output  [22:0] p_0_0_7_0_0_079_out;
output   p_0_0_7_0_0_079_out_ap_vld;
output  [22:0] p_0_0_6_0_0_077_out;
output   p_0_0_6_0_0_077_out_ap_vld;
output  [22:0] p_0_0_5_0_0_075_out;
output   p_0_0_5_0_0_075_out_ap_vld;
output  [22:0] p_0_0_4_0_0_073_out;
output   p_0_0_4_0_0_073_out_ap_vld;
output  [22:0] p_0_0_3_0_0_071_out;
output   p_0_0_3_0_0_071_out_ap_vld;
output  [22:0] p_0_0_2_0_0_069_out;
output   p_0_0_2_0_0_069_out_ap_vld;
output  [22:0] p_0_0_1_0_0_067_out;
output   p_0_0_1_0_0_067_out_ap_vld;
output  [22:0] p_0_0_0_0_0_065_out;
output   p_0_0_0_0_0_065_out_ap_vld;
output  [22:0] p_0_0_7_0_0_063_out;
output   p_0_0_7_0_0_063_out_ap_vld;
output  [22:0] p_0_0_6_0_0_061_out;
output   p_0_0_6_0_0_061_out_ap_vld;
output  [22:0] p_0_0_5_0_0_059_out;
output   p_0_0_5_0_0_059_out_ap_vld;
output  [22:0] p_0_0_4_0_0_057_out;
output   p_0_0_4_0_0_057_out_ap_vld;
output  [22:0] p_0_0_3_0_0_055_out;
output   p_0_0_3_0_0_055_out_ap_vld;
output  [22:0] p_0_0_2_0_0_053_out;
output   p_0_0_2_0_0_053_out_ap_vld;
output  [22:0] p_0_0_1_0_0_051_out;
output   p_0_0_1_0_0_051_out_ap_vld;
output  [22:0] p_0_0_0_0_0_049_out;
output   p_0_0_0_0_0_049_out_ap_vld;
output  [22:0] p_0_0_7_0_0_047_out;
output   p_0_0_7_0_0_047_out_ap_vld;
output  [22:0] p_0_0_6_0_0_045_out;
output   p_0_0_6_0_0_045_out_ap_vld;
output  [22:0] p_0_0_5_0_0_043_out;
output   p_0_0_5_0_0_043_out_ap_vld;
output  [22:0] p_0_0_4_0_0_041_out;
output   p_0_0_4_0_0_041_out_ap_vld;
output  [22:0] p_0_0_3_0_0_039_out;
output   p_0_0_3_0_0_039_out_ap_vld;
output  [22:0] p_0_0_2_0_0_037_out;
output   p_0_0_2_0_0_037_out_ap_vld;
output  [22:0] p_0_0_1_0_0_035_out;
output   p_0_0_1_0_0_035_out_ap_vld;
output  [22:0] p_0_0_0_0_0_033_out;
output   p_0_0_0_0_0_033_out_ap_vld;
output  [22:0] p_0_0_7_0_0_031_out;
output   p_0_0_7_0_0_031_out_ap_vld;
output  [22:0] p_0_0_6_0_0_029_out;
output   p_0_0_6_0_0_029_out_ap_vld;
output  [22:0] p_0_0_5_0_0_027_out;
output   p_0_0_5_0_0_027_out_ap_vld;
output  [22:0] p_0_0_4_0_0_025_out;
output   p_0_0_4_0_0_025_out_ap_vld;
output  [22:0] p_0_0_3_0_0_023_out;
output   p_0_0_3_0_0_023_out_ap_vld;
output  [22:0] p_0_0_2_0_0_021_out;
output   p_0_0_2_0_0_021_out_ap_vld;
output  [22:0] p_0_0_1_0_0_019_out;
output   p_0_0_1_0_0_019_out_ap_vld;
output  [22:0] p_0_0_0_0_0_017_out;
output   p_0_0_0_0_0_017_out_ap_vld;
output  [22:0] p_0_0_7_0_0_015_out;
output   p_0_0_7_0_0_015_out_ap_vld;
output  [22:0] p_0_0_6_0_0_013_out;
output   p_0_0_6_0_0_013_out_ap_vld;
output  [22:0] p_0_0_5_0_0_011_out;
output   p_0_0_5_0_0_011_out_ap_vld;
output  [22:0] p_0_0_4_0_0_09_out;
output   p_0_0_4_0_0_09_out_ap_vld;
output  [22:0] p_0_0_3_0_0_07_out;
output   p_0_0_3_0_0_07_out_ap_vld;
output  [22:0] p_0_0_2_0_0_05_out;
output   p_0_0_2_0_0_05_out_ap_vld;
output  [22:0] p_0_0_1_0_0_03_out;
output   p_0_0_1_0_0_03_out_ap_vld;
output  [22:0] p_0_0_0_0_0_01_out;
output   p_0_0_0_0_0_01_out_ap_vld;

reg ap_idle;
reg v_stream_TREADY;
reg p_0_0_7_0_0_0127_out_ap_vld;
reg p_0_0_6_0_0_0125_out_ap_vld;
reg p_0_0_5_0_0_0123_out_ap_vld;
reg p_0_0_4_0_0_0121_out_ap_vld;
reg p_0_0_3_0_0_0119_out_ap_vld;
reg p_0_0_2_0_0_0117_out_ap_vld;
reg p_0_0_1_0_0_0115_out_ap_vld;
reg p_0_0_0_0_0_0113_out_ap_vld;
reg p_0_0_7_0_0_0111_out_ap_vld;
reg p_0_0_6_0_0_0109_out_ap_vld;
reg p_0_0_5_0_0_0107_out_ap_vld;
reg p_0_0_4_0_0_0105_out_ap_vld;
reg p_0_0_3_0_0_0103_out_ap_vld;
reg p_0_0_2_0_0_0101_out_ap_vld;
reg p_0_0_1_0_0_099_out_ap_vld;
reg p_0_0_0_0_0_097_out_ap_vld;
reg p_0_0_7_0_0_095_out_ap_vld;
reg p_0_0_6_0_0_093_out_ap_vld;
reg p_0_0_5_0_0_091_out_ap_vld;
reg p_0_0_4_0_0_089_out_ap_vld;
reg p_0_0_3_0_0_087_out_ap_vld;
reg p_0_0_2_0_0_085_out_ap_vld;
reg p_0_0_1_0_0_083_out_ap_vld;
reg p_0_0_0_0_0_081_out_ap_vld;
reg p_0_0_7_0_0_079_out_ap_vld;
reg p_0_0_6_0_0_077_out_ap_vld;
reg p_0_0_5_0_0_075_out_ap_vld;
reg p_0_0_4_0_0_073_out_ap_vld;
reg p_0_0_3_0_0_071_out_ap_vld;
reg p_0_0_2_0_0_069_out_ap_vld;
reg p_0_0_1_0_0_067_out_ap_vld;
reg p_0_0_0_0_0_065_out_ap_vld;
reg p_0_0_7_0_0_063_out_ap_vld;
reg p_0_0_6_0_0_061_out_ap_vld;
reg p_0_0_5_0_0_059_out_ap_vld;
reg p_0_0_4_0_0_057_out_ap_vld;
reg p_0_0_3_0_0_055_out_ap_vld;
reg p_0_0_2_0_0_053_out_ap_vld;
reg p_0_0_1_0_0_051_out_ap_vld;
reg p_0_0_0_0_0_049_out_ap_vld;
reg p_0_0_7_0_0_047_out_ap_vld;
reg p_0_0_6_0_0_045_out_ap_vld;
reg p_0_0_5_0_0_043_out_ap_vld;
reg p_0_0_4_0_0_041_out_ap_vld;
reg p_0_0_3_0_0_039_out_ap_vld;
reg p_0_0_2_0_0_037_out_ap_vld;
reg p_0_0_1_0_0_035_out_ap_vld;
reg p_0_0_0_0_0_033_out_ap_vld;
reg p_0_0_7_0_0_031_out_ap_vld;
reg p_0_0_6_0_0_029_out_ap_vld;
reg p_0_0_5_0_0_027_out_ap_vld;
reg p_0_0_4_0_0_025_out_ap_vld;
reg p_0_0_3_0_0_023_out_ap_vld;
reg p_0_0_2_0_0_021_out_ap_vld;
reg p_0_0_1_0_0_019_out_ap_vld;
reg p_0_0_0_0_0_017_out_ap_vld;
reg p_0_0_7_0_0_015_out_ap_vld;
reg p_0_0_6_0_0_013_out_ap_vld;
reg p_0_0_5_0_0_011_out_ap_vld;
reg p_0_0_4_0_0_09_out_ap_vld;
reg p_0_0_3_0_0_07_out_ap_vld;
reg p_0_0_2_0_0_05_out_ap_vld;
reg p_0_0_1_0_0_03_out_ap_vld;
reg p_0_0_0_0_0_01_out_ap_vld;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [0:0] icmp_ln226_fu_930_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    v_stream_TDATA_blk_n;
reg   [3:0] ti_fu_208;
wire   [3:0] add_ln226_fu_936_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_ti_1;
reg   [22:0] p_0_0_0_0_0_01_fu_212;
wire   [22:0] trunc_ln228_fu_946_p1;
wire   [2:0] trunc_ln226_fu_942_p1;
reg   [22:0] p_0_0_1_0_0_03_fu_216;
reg   [22:0] p_0_0_2_0_0_05_fu_220;
reg   [22:0] p_0_0_3_0_0_07_fu_224;
reg   [22:0] p_0_0_4_0_0_09_fu_228;
reg   [22:0] p_0_0_5_0_0_011_fu_232;
reg   [22:0] p_0_0_6_0_0_013_fu_236;
reg   [22:0] p_0_0_7_0_0_015_fu_240;
reg   [22:0] p_0_0_0_0_0_017_fu_244;
reg   [22:0] p_0_0_1_0_0_019_fu_248;
reg   [22:0] p_0_0_2_0_0_021_fu_252;
reg   [22:0] p_0_0_3_0_0_023_fu_256;
reg   [22:0] p_0_0_4_0_0_025_fu_260;
reg   [22:0] p_0_0_5_0_0_027_fu_264;
reg   [22:0] p_0_0_6_0_0_029_fu_268;
reg   [22:0] p_0_0_7_0_0_031_fu_272;
reg   [22:0] p_0_0_0_0_0_033_fu_276;
reg   [22:0] p_0_0_1_0_0_035_fu_280;
reg   [22:0] p_0_0_2_0_0_037_fu_284;
reg   [22:0] p_0_0_3_0_0_039_fu_288;
reg   [22:0] p_0_0_4_0_0_041_fu_292;
reg   [22:0] p_0_0_5_0_0_043_fu_296;
reg   [22:0] p_0_0_6_0_0_045_fu_300;
reg   [22:0] p_0_0_7_0_0_047_fu_304;
reg   [22:0] p_0_0_0_0_0_049_fu_308;
reg   [22:0] p_0_0_1_0_0_051_fu_312;
reg   [22:0] p_0_0_2_0_0_053_fu_316;
reg   [22:0] p_0_0_3_0_0_055_fu_320;
reg   [22:0] p_0_0_4_0_0_057_fu_324;
reg   [22:0] p_0_0_5_0_0_059_fu_328;
reg   [22:0] p_0_0_6_0_0_061_fu_332;
reg   [22:0] p_0_0_7_0_0_063_fu_336;
reg   [22:0] p_0_0_0_0_0_065_fu_340;
reg   [22:0] p_0_0_1_0_0_067_fu_344;
reg   [22:0] p_0_0_2_0_0_069_fu_348;
reg   [22:0] p_0_0_3_0_0_071_fu_352;
reg   [22:0] p_0_0_4_0_0_073_fu_356;
reg   [22:0] p_0_0_5_0_0_075_fu_360;
reg   [22:0] p_0_0_6_0_0_077_fu_364;
reg   [22:0] p_0_0_7_0_0_079_fu_368;
reg   [22:0] p_0_0_0_0_0_081_fu_372;
reg   [22:0] p_0_0_1_0_0_083_fu_376;
reg   [22:0] p_0_0_2_0_0_085_fu_380;
reg   [22:0] p_0_0_3_0_0_087_fu_384;
reg   [22:0] p_0_0_4_0_0_089_fu_388;
reg   [22:0] p_0_0_5_0_0_091_fu_392;
reg   [22:0] p_0_0_6_0_0_093_fu_396;
reg   [22:0] p_0_0_7_0_0_095_fu_400;
reg   [22:0] p_0_0_0_0_0_097_fu_404;
reg   [22:0] p_0_0_1_0_0_099_fu_408;
reg   [22:0] p_0_0_2_0_0_0101_fu_412;
reg   [22:0] p_0_0_3_0_0_0103_fu_416;
reg   [22:0] p_0_0_4_0_0_0105_fu_420;
reg   [22:0] p_0_0_5_0_0_0107_fu_424;
reg   [22:0] p_0_0_6_0_0_0109_fu_428;
reg   [22:0] p_0_0_7_0_0_0111_fu_432;
reg   [22:0] p_0_0_0_0_0_0113_fu_436;
reg   [22:0] p_0_0_1_0_0_0115_fu_440;
reg   [22:0] p_0_0_2_0_0_0117_fu_444;
reg   [22:0] p_0_0_3_0_0_0119_fu_448;
reg   [22:0] p_0_0_4_0_0_0121_fu_452;
reg   [22:0] p_0_0_5_0_0_0123_fu_456;
reg   [22:0] p_0_0_6_0_0_0125_fu_460;
reg   [22:0] p_0_0_7_0_0_0127_fu_464;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [0:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 1'd1;
//#0 ti_fu_208 = 4'd0;
//#0 p_0_0_0_0_0_01_fu_212 = 23'd0;
//#0 p_0_0_1_0_0_03_fu_216 = 23'd0;
//#0 p_0_0_2_0_0_05_fu_220 = 23'd0;
//#0 p_0_0_3_0_0_07_fu_224 = 23'd0;
//#0 p_0_0_4_0_0_09_fu_228 = 23'd0;
//#0 p_0_0_5_0_0_011_fu_232 = 23'd0;
//#0 p_0_0_6_0_0_013_fu_236 = 23'd0;
//#0 p_0_0_7_0_0_015_fu_240 = 23'd0;
//#0 p_0_0_0_0_0_017_fu_244 = 23'd0;
//#0 p_0_0_1_0_0_019_fu_248 = 23'd0;
//#0 p_0_0_2_0_0_021_fu_252 = 23'd0;
//#0 p_0_0_3_0_0_023_fu_256 = 23'd0;
//#0 p_0_0_4_0_0_025_fu_260 = 23'd0;
//#0 p_0_0_5_0_0_027_fu_264 = 23'd0;
//#0 p_0_0_6_0_0_029_fu_268 = 23'd0;
//#0 p_0_0_7_0_0_031_fu_272 = 23'd0;
//#0 p_0_0_0_0_0_033_fu_276 = 23'd0;
//#0 p_0_0_1_0_0_035_fu_280 = 23'd0;
//#0 p_0_0_2_0_0_037_fu_284 = 23'd0;
//#0 p_0_0_3_0_0_039_fu_288 = 23'd0;
//#0 p_0_0_4_0_0_041_fu_292 = 23'd0;
//#0 p_0_0_5_0_0_043_fu_296 = 23'd0;
//#0 p_0_0_6_0_0_045_fu_300 = 23'd0;
//#0 p_0_0_7_0_0_047_fu_304 = 23'd0;
//#0 p_0_0_0_0_0_049_fu_308 = 23'd0;
//#0 p_0_0_1_0_0_051_fu_312 = 23'd0;
//#0 p_0_0_2_0_0_053_fu_316 = 23'd0;
//#0 p_0_0_3_0_0_055_fu_320 = 23'd0;
//#0 p_0_0_4_0_0_057_fu_324 = 23'd0;
//#0 p_0_0_5_0_0_059_fu_328 = 23'd0;
//#0 p_0_0_6_0_0_061_fu_332 = 23'd0;
//#0 p_0_0_7_0_0_063_fu_336 = 23'd0;
//#0 p_0_0_0_0_0_065_fu_340 = 23'd0;
//#0 p_0_0_1_0_0_067_fu_344 = 23'd0;
//#0 p_0_0_2_0_0_069_fu_348 = 23'd0;
//#0 p_0_0_3_0_0_071_fu_352 = 23'd0;
//#0 p_0_0_4_0_0_073_fu_356 = 23'd0;
//#0 p_0_0_5_0_0_075_fu_360 = 23'd0;
//#0 p_0_0_6_0_0_077_fu_364 = 23'd0;
//#0 p_0_0_7_0_0_079_fu_368 = 23'd0;
//#0 p_0_0_0_0_0_081_fu_372 = 23'd0;
//#0 p_0_0_1_0_0_083_fu_376 = 23'd0;
//#0 p_0_0_2_0_0_085_fu_380 = 23'd0;
//#0 p_0_0_3_0_0_087_fu_384 = 23'd0;
//#0 p_0_0_4_0_0_089_fu_388 = 23'd0;
//#0 p_0_0_5_0_0_091_fu_392 = 23'd0;
//#0 p_0_0_6_0_0_093_fu_396 = 23'd0;
//#0 p_0_0_7_0_0_095_fu_400 = 23'd0;
//#0 p_0_0_0_0_0_097_fu_404 = 23'd0;
//#0 p_0_0_1_0_0_099_fu_408 = 23'd0;
//#0 p_0_0_2_0_0_0101_fu_412 = 23'd0;
//#0 p_0_0_3_0_0_0103_fu_416 = 23'd0;
//#0 p_0_0_4_0_0_0105_fu_420 = 23'd0;
//#0 p_0_0_5_0_0_0107_fu_424 = 23'd0;
//#0 p_0_0_6_0_0_0109_fu_428 = 23'd0;
//#0 p_0_0_7_0_0_0111_fu_432 = 23'd0;
//#0 p_0_0_0_0_0_0113_fu_436 = 23'd0;
//#0 p_0_0_1_0_0_0115_fu_440 = 23'd0;
//#0 p_0_0_2_0_0_0117_fu_444 = 23'd0;
//#0 p_0_0_3_0_0_0119_fu_448 = 23'd0;
//#0 p_0_0_4_0_0_0121_fu_452 = 23'd0;
//#0 p_0_0_5_0_0_0123_fu_456 = 23'd0;
//#0 p_0_0_6_0_0_0125_fu_460 = 23'd0;
//#0 p_0_0_7_0_0_0127_fu_464 = 23'd0;
//#0 ap_done_reg = 1'b0;
end

RV_GEMM_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
        if ((icmp_ln226_fu_930_p2 == 1'd0)) begin
            ti_fu_208 <= add_ln226_fu_936_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            ti_fu_208 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd7))) begin
        p_0_0_0_0_0_0113_fu_436 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_0115_fu_440 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0117_fu_444 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0119_fu_448 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0121_fu_452 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0123_fu_456 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0125_fu_460 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0127_fu_464 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd5))) begin
        p_0_0_0_0_0_017_fu_244 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_019_fu_248 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_021_fu_252 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_023_fu_256 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_025_fu_260 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_027_fu_264 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_029_fu_268 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_031_fu_272 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd6))) begin
        p_0_0_0_0_0_01_fu_212 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_03_fu_216 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_05_fu_220 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_07_fu_224 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_09_fu_228 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_011_fu_232 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_013_fu_236 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_015_fu_240 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd4))) begin
        p_0_0_0_0_0_033_fu_276 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_035_fu_280 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_037_fu_284 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_039_fu_288 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_041_fu_292 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_043_fu_296 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_045_fu_300 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_047_fu_304 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd3))) begin
        p_0_0_0_0_0_049_fu_308 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_051_fu_312 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_053_fu_316 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_055_fu_320 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_057_fu_324 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_059_fu_328 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_061_fu_332 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_063_fu_336 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd2))) begin
        p_0_0_0_0_0_065_fu_340 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_067_fu_344 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_069_fu_348 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_071_fu_352 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_073_fu_356 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_075_fu_360 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_077_fu_364 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_079_fu_368 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd1))) begin
        p_0_0_0_0_0_081_fu_372 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_083_fu_376 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_085_fu_380 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_087_fu_384 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_089_fu_388 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_091_fu_392 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_093_fu_396 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_095_fu_400 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (trunc_ln226_fu_942_p1 == 3'd0))) begin
        p_0_0_0_0_0_097_fu_404 <= trunc_ln228_fu_946_p1;
        p_0_0_1_0_0_099_fu_408 <= {{v_stream_TDATA[45:23]}};
        p_0_0_2_0_0_0101_fu_412 <= {{v_stream_TDATA[68:46]}};
        p_0_0_3_0_0_0103_fu_416 <= {{v_stream_TDATA[91:69]}};
        p_0_0_4_0_0_0105_fu_420 <= {{v_stream_TDATA[114:92]}};
        p_0_0_5_0_0_0107_fu_424 <= {{v_stream_TDATA[137:115]}};
        p_0_0_6_0_0_0109_fu_428 <= {{v_stream_TDATA[160:138]}};
        p_0_0_7_0_0_0111_fu_432 <= {{v_stream_TDATA[183:161]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_ti_1 = 4'd0;
    end else begin
        ap_sig_allocacmp_ti_1 = ti_fu_208;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_0113_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_0113_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_017_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_017_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_01_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_01_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_033_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_033_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_049_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_049_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_065_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_065_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_081_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_081_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_0_0_0_097_out_ap_vld = 1'b1;
    end else begin
        p_0_0_0_0_0_097_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_0115_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_0115_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_019_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_019_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_035_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_035_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_03_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_03_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_051_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_051_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_067_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_067_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_083_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_083_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_1_0_0_099_out_ap_vld = 1'b1;
    end else begin
        p_0_0_1_0_0_099_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0101_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0101_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_0117_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_0117_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_021_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_021_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_037_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_037_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_053_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_053_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_05_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_05_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_069_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_069_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_2_0_0_085_out_ap_vld = 1'b1;
    end else begin
        p_0_0_2_0_0_085_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0103_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0103_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_0119_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_0119_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_023_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_023_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_039_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_039_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_055_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_055_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_071_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_071_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_07_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_07_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_3_0_0_087_out_ap_vld = 1'b1;
    end else begin
        p_0_0_3_0_0_087_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0105_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0105_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_0121_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_0121_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_025_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_025_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_041_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_041_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_057_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_057_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_073_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_073_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_089_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_089_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_4_0_0_09_out_ap_vld = 1'b1;
    end else begin
        p_0_0_4_0_0_09_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0107_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0107_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_011_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_011_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_0123_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_0123_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_027_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_027_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_043_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_043_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_059_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_059_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_075_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_075_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_5_0_0_091_out_ap_vld = 1'b1;
    end else begin
        p_0_0_5_0_0_091_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0109_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0109_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_0125_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_0125_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_013_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_013_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_029_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_029_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_045_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_045_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_061_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_061_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_077_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_077_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_6_0_0_093_out_ap_vld = 1'b1;
    end else begin
        p_0_0_6_0_0_093_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0111_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0111_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_0127_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_0127_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_015_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_015_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_031_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_031_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_047_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_047_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_063_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_063_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_079_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_079_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
        p_0_0_7_0_0_095_out_ap_vld = 1'b1;
    end else begin
        p_0_0_7_0_0_095_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1))) begin
        v_stream_TDATA_blk_n = v_stream_TVALID;
    end else begin
        v_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (icmp_ln226_fu_930_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1))) begin
        v_stream_TREADY = 1'b1;
    end else begin
        v_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln226_fu_936_p2 = (ap_sig_allocacmp_ti_1 + 4'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln226_fu_930_p2 == 1'd0) & (v_stream_TVALID == 1'b0)));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln226_fu_930_p2 = ((ap_sig_allocacmp_ti_1 == 4'd8) ? 1'b1 : 1'b0);

assign p_0_0_0_0_0_0113_out = p_0_0_0_0_0_0113_fu_436;

assign p_0_0_0_0_0_017_out = p_0_0_0_0_0_017_fu_244;

assign p_0_0_0_0_0_01_out = p_0_0_0_0_0_01_fu_212;

assign p_0_0_0_0_0_033_out = p_0_0_0_0_0_033_fu_276;

assign p_0_0_0_0_0_049_out = p_0_0_0_0_0_049_fu_308;

assign p_0_0_0_0_0_065_out = p_0_0_0_0_0_065_fu_340;

assign p_0_0_0_0_0_081_out = p_0_0_0_0_0_081_fu_372;

assign p_0_0_0_0_0_097_out = p_0_0_0_0_0_097_fu_404;

assign p_0_0_1_0_0_0115_out = p_0_0_1_0_0_0115_fu_440;

assign p_0_0_1_0_0_019_out = p_0_0_1_0_0_019_fu_248;

assign p_0_0_1_0_0_035_out = p_0_0_1_0_0_035_fu_280;

assign p_0_0_1_0_0_03_out = p_0_0_1_0_0_03_fu_216;

assign p_0_0_1_0_0_051_out = p_0_0_1_0_0_051_fu_312;

assign p_0_0_1_0_0_067_out = p_0_0_1_0_0_067_fu_344;

assign p_0_0_1_0_0_083_out = p_0_0_1_0_0_083_fu_376;

assign p_0_0_1_0_0_099_out = p_0_0_1_0_0_099_fu_408;

assign p_0_0_2_0_0_0101_out = p_0_0_2_0_0_0101_fu_412;

assign p_0_0_2_0_0_0117_out = p_0_0_2_0_0_0117_fu_444;

assign p_0_0_2_0_0_021_out = p_0_0_2_0_0_021_fu_252;

assign p_0_0_2_0_0_037_out = p_0_0_2_0_0_037_fu_284;

assign p_0_0_2_0_0_053_out = p_0_0_2_0_0_053_fu_316;

assign p_0_0_2_0_0_05_out = p_0_0_2_0_0_05_fu_220;

assign p_0_0_2_0_0_069_out = p_0_0_2_0_0_069_fu_348;

assign p_0_0_2_0_0_085_out = p_0_0_2_0_0_085_fu_380;

assign p_0_0_3_0_0_0103_out = p_0_0_3_0_0_0103_fu_416;

assign p_0_0_3_0_0_0119_out = p_0_0_3_0_0_0119_fu_448;

assign p_0_0_3_0_0_023_out = p_0_0_3_0_0_023_fu_256;

assign p_0_0_3_0_0_039_out = p_0_0_3_0_0_039_fu_288;

assign p_0_0_3_0_0_055_out = p_0_0_3_0_0_055_fu_320;

assign p_0_0_3_0_0_071_out = p_0_0_3_0_0_071_fu_352;

assign p_0_0_3_0_0_07_out = p_0_0_3_0_0_07_fu_224;

assign p_0_0_3_0_0_087_out = p_0_0_3_0_0_087_fu_384;

assign p_0_0_4_0_0_0105_out = p_0_0_4_0_0_0105_fu_420;

assign p_0_0_4_0_0_0121_out = p_0_0_4_0_0_0121_fu_452;

assign p_0_0_4_0_0_025_out = p_0_0_4_0_0_025_fu_260;

assign p_0_0_4_0_0_041_out = p_0_0_4_0_0_041_fu_292;

assign p_0_0_4_0_0_057_out = p_0_0_4_0_0_057_fu_324;

assign p_0_0_4_0_0_073_out = p_0_0_4_0_0_073_fu_356;

assign p_0_0_4_0_0_089_out = p_0_0_4_0_0_089_fu_388;

assign p_0_0_4_0_0_09_out = p_0_0_4_0_0_09_fu_228;

assign p_0_0_5_0_0_0107_out = p_0_0_5_0_0_0107_fu_424;

assign p_0_0_5_0_0_011_out = p_0_0_5_0_0_011_fu_232;

assign p_0_0_5_0_0_0123_out = p_0_0_5_0_0_0123_fu_456;

assign p_0_0_5_0_0_027_out = p_0_0_5_0_0_027_fu_264;

assign p_0_0_5_0_0_043_out = p_0_0_5_0_0_043_fu_296;

assign p_0_0_5_0_0_059_out = p_0_0_5_0_0_059_fu_328;

assign p_0_0_5_0_0_075_out = p_0_0_5_0_0_075_fu_360;

assign p_0_0_5_0_0_091_out = p_0_0_5_0_0_091_fu_392;

assign p_0_0_6_0_0_0109_out = p_0_0_6_0_0_0109_fu_428;

assign p_0_0_6_0_0_0125_out = p_0_0_6_0_0_0125_fu_460;

assign p_0_0_6_0_0_013_out = p_0_0_6_0_0_013_fu_236;

assign p_0_0_6_0_0_029_out = p_0_0_6_0_0_029_fu_268;

assign p_0_0_6_0_0_045_out = p_0_0_6_0_0_045_fu_300;

assign p_0_0_6_0_0_061_out = p_0_0_6_0_0_061_fu_332;

assign p_0_0_6_0_0_077_out = p_0_0_6_0_0_077_fu_364;

assign p_0_0_6_0_0_093_out = p_0_0_6_0_0_093_fu_396;

assign p_0_0_7_0_0_0111_out = p_0_0_7_0_0_0111_fu_432;

assign p_0_0_7_0_0_0127_out = p_0_0_7_0_0_0127_fu_464;

assign p_0_0_7_0_0_015_out = p_0_0_7_0_0_015_fu_240;

assign p_0_0_7_0_0_031_out = p_0_0_7_0_0_031_fu_272;

assign p_0_0_7_0_0_047_out = p_0_0_7_0_0_047_fu_304;

assign p_0_0_7_0_0_063_out = p_0_0_7_0_0_063_fu_336;

assign p_0_0_7_0_0_079_out = p_0_0_7_0_0_079_fu_368;

assign p_0_0_7_0_0_095_out = p_0_0_7_0_0_095_fu_400;

assign trunc_ln226_fu_942_p1 = ap_sig_allocacmp_ti_1[2:0];

assign trunc_ln228_fu_946_p1 = v_stream_TDATA[22:0];

endmodule //RV_GEMM_load_quantize_vit_v_Pipeline_VITIS_LOOP_226_3
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_quantize_a_vec_Pipeline_VITIS_LOOP_15_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        x_4,
        i_out,
        i_out_ap_vld,
        ap_return
);

parameter    ap_ST_fsm_state1 = 5'd1;
parameter    ap_ST_fsm_state2 = 5'd2;
parameter    ap_ST_fsm_state3 = 5'd4;
parameter    ap_ST_fsm_state4 = 5'd8;
parameter    ap_ST_fsm_state5 = 5'd16;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [22:0] x_4;
output  [4:0] i_out;
output   i_out_ap_vld;
output  [0:0] ap_return;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg[4:0] i_out;
reg i_out_ap_vld;
reg[0:0] ap_return;

(* fsm_encoding = "none" *) reg   [4:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [4:0] i_2_reg_130;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln15_fu_82_p2;
reg   [0:0] icmp_ln15_reg_135;
wire   [4:0] i_3_fu_88_p2;
reg   [4:0] i_3_reg_139;
wire   [22:0] shl_ln16_fu_98_p2;
reg   [22:0] shl_ln16_reg_144;
reg   [0:0] ap_phi_mux_UnifiedRetVal_phi_fu_66_p4;
reg   [0:0] UnifiedRetVal_reg_62;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state4;
reg   [4:0] i_fu_44;
wire    ap_CS_fsm_state3;
wire   [0:0] icmp_ln16_fu_108_p2;
wire   [22:0] zext_ln16_fu_94_p1;
wire   [22:0] and_ln16_fu_104_p2;
reg   [0:0] ap_return_preg;
reg   [4:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 5'd1;
//#0 i_fu_44 = 5'd0;
//#0 ap_return_preg = 1'd0;
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_return_preg <= 1'd0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state4)) begin
            ap_return_preg <= ap_phi_mux_UnifiedRetVal_phi_fu_66_p4;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln15_reg_135 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        UnifiedRetVal_reg_62 <= 1'd0;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        UnifiedRetVal_reg_62 <= 1'd1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        i_fu_44 <= 5'd23;
    end else if (((icmp_ln16_fu_108_p2 == 1'd1) & (icmp_ln15_reg_135 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        i_fu_44 <= i_3_reg_139;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        i_2_reg_130 <= i_fu_44;
        i_3_reg_139 <= i_3_fu_88_p2;
        icmp_ln15_reg_135 <= icmp_ln15_fu_82_p2;
        shl_ln16_reg_144 <= shl_ln16_fu_98_p2;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
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
    if (((icmp_ln15_reg_135 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        ap_phi_mux_UnifiedRetVal_phi_fu_66_p4 = 1'd0;
    end else begin
        ap_phi_mux_UnifiedRetVal_phi_fu_66_p4 = UnifiedRetVal_reg_62;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        ap_return = ap_phi_mux_UnifiedRetVal_phi_fu_66_p4;
    end else begin
        ap_return = ap_return_preg;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        i_out = 5'd0;
    end else if (((icmp_ln15_reg_135 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        i_out = i_2_reg_130;
    end else begin
        i_out = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state5) | ((icmp_ln15_reg_135 == 1'd0) & (1'b1 == ap_CS_fsm_state4)))) begin
        i_out_ap_vld = 1'b1;
    end else begin
        i_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((icmp_ln16_fu_108_p2 == 1'd1) & (icmp_ln15_reg_135 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else if (((icmp_ln15_reg_135 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign and_ln16_fu_104_p2 = (x_4 & shl_ln16_reg_144);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign i_3_fu_88_p2 = ($signed(i_fu_44) + $signed(5'd31));

assign icmp_ln15_fu_82_p2 = ((i_fu_44 == 5'd0) ? 1'b1 : 1'b0);

assign icmp_ln16_fu_108_p2 = ((and_ln16_fu_104_p2 == 23'd0) ? 1'b1 : 1'b0);

assign shl_ln16_fu_98_p2 = 23'd1 << zext_ln16_fu_94_p1;

assign zext_ln16_fu_94_p1 = i_3_fu_88_p2;

endmodule //RV_GEMM_quantize_a_vec_Pipeline_VITIS_LOOP_15_1
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1 ns / 1 ps

  (* use_dsp = "yes" *)  module RV_GEMM_mul_8s_8s_16_1_1(din0, din1, dout);
parameter ID = 1;
parameter NUM_STAGE = 0;
parameter din0_WIDTH = 14;
parameter din1_WIDTH = 12;
parameter dout_WIDTH = 26;

input [din0_WIDTH - 1 : 0] din0; 
input [din1_WIDTH - 1 : 0] din1; 
output [dout_WIDTH - 1 : 0] dout;

wire signed [dout_WIDTH - 1 : 0] tmp_product;













assign tmp_product = $signed(din0) * $signed(din1);








assign dout = tmp_product;







endmodule
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        valid_st,
        zext_ln350,
        vq_buf_0_address0,
        vq_buf_0_ce0,
        vq_buf_0_q0,
        vq_buf_1_address0,
        vq_buf_1_ce0,
        vq_buf_1_q0,
        vq_buf_2_address0,
        vq_buf_2_ce0,
        vq_buf_2_q0,
        vq_buf_3_address0,
        vq_buf_3_ce0,
        vq_buf_3_q0,
        vq_buf_4_address0,
        vq_buf_4_ce0,
        vq_buf_4_q0,
        vq_buf_5_address0,
        vq_buf_5_ce0,
        vq_buf_5_q0,
        vq_buf_6_address0,
        vq_buf_6_ce0,
        vq_buf_6_q0,
        vq_buf_7_address0,
        vq_buf_7_ce0,
        vq_buf_7_q0,
        vs_buf_address0,
        vs_buf_ce0,
        vs_buf_q0,
        rq_buf_address0,
        rq_buf_ce0,
        rq_buf_q0,
        rq_buf_1_address0,
        rq_buf_1_ce0,
        rq_buf_1_q0,
        rq_buf_2_address0,
        rq_buf_2_ce0,
        rq_buf_2_q0,
        rq_buf_3_address0,
        rq_buf_3_ce0,
        rq_buf_3_q0,
        rq_buf_4_address0,
        rq_buf_4_ce0,
        rq_buf_4_q0,
        rq_buf_5_address0,
        rq_buf_5_ce0,
        rq_buf_5_q0,
        rq_buf_6_address0,
        rq_buf_6_ce0,
        rq_buf_6_q0,
        rq_buf_7_address0,
        rq_buf_7_ce0,
        rq_buf_7_q0,
        rs_buf_address0,
        rs_buf_ce0,
        rs_buf_q0,
        psum_vec_7_out,
        psum_vec_7_out_ap_vld,
        psum_vec_6_out,
        psum_vec_6_out_ap_vld,
        psum_vec_5_out,
        psum_vec_5_out_ap_vld,
        psum_vec_4_out,
        psum_vec_4_out_ap_vld,
        psum_vec_3_out,
        psum_vec_3_out_ap_vld,
        psum_vec_2_out,
        psum_vec_2_out_ap_vld,
        psum_vec_1_out,
        psum_vec_1_out_ap_vld,
        psum_vec_out,
        psum_vec_out_ap_vld
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
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;
parameter    ap_ST_iter6_fsm_state0 = 2'd1;
parameter    ap_ST_iter7_fsm_state0 = 2'd1;
parameter    ap_ST_iter8_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [7:0] valid_st;
input  [9:0] zext_ln350;
output  [9:0] vq_buf_0_address0;
output   vq_buf_0_ce0;
input  [63:0] vq_buf_0_q0;
output  [9:0] vq_buf_1_address0;
output   vq_buf_1_ce0;
input  [63:0] vq_buf_1_q0;
output  [9:0] vq_buf_2_address0;
output   vq_buf_2_ce0;
input  [63:0] vq_buf_2_q0;
output  [9:0] vq_buf_3_address0;
output   vq_buf_3_ce0;
input  [63:0] vq_buf_3_q0;
output  [9:0] vq_buf_4_address0;
output   vq_buf_4_ce0;
input  [63:0] vq_buf_4_q0;
output  [9:0] vq_buf_5_address0;
output   vq_buf_5_ce0;
input  [63:0] vq_buf_5_q0;
output  [9:0] vq_buf_6_address0;
output   vq_buf_6_ce0;
input  [63:0] vq_buf_6_q0;
output  [9:0] vq_buf_7_address0;
output   vq_buf_7_ce0;
input  [63:0] vq_buf_7_q0;
output  [9:0] vs_buf_address0;
output   vs_buf_ce0;
input  [31:0] vs_buf_q0;
output  [6:0] rq_buf_address0;
output   rq_buf_ce0;
input  [7:0] rq_buf_q0;
output  [6:0] rq_buf_1_address0;
output   rq_buf_1_ce0;
input  [7:0] rq_buf_1_q0;
output  [6:0] rq_buf_2_address0;
output   rq_buf_2_ce0;
input  [7:0] rq_buf_2_q0;
output  [6:0] rq_buf_3_address0;
output   rq_buf_3_ce0;
input  [7:0] rq_buf_3_q0;
output  [6:0] rq_buf_4_address0;
output   rq_buf_4_ce0;
input  [7:0] rq_buf_4_q0;
output  [6:0] rq_buf_5_address0;
output   rq_buf_5_ce0;
input  [7:0] rq_buf_5_q0;
output  [6:0] rq_buf_6_address0;
output   rq_buf_6_ce0;
input  [7:0] rq_buf_6_q0;
output  [6:0] rq_buf_7_address0;
output   rq_buf_7_ce0;
input  [7:0] rq_buf_7_q0;
output  [6:0] rs_buf_address0;
output   rs_buf_ce0;
input  [3:0] rs_buf_q0;
output  [39:0] psum_vec_7_out;
output   psum_vec_7_out_ap_vld;
output  [39:0] psum_vec_6_out;
output   psum_vec_6_out_ap_vld;
output  [39:0] psum_vec_5_out;
output   psum_vec_5_out_ap_vld;
output  [39:0] psum_vec_4_out;
output   psum_vec_4_out_ap_vld;
output  [39:0] psum_vec_3_out;
output   psum_vec_3_out_ap_vld;
output  [39:0] psum_vec_2_out;
output   psum_vec_2_out_ap_vld;
output  [39:0] psum_vec_1_out;
output   psum_vec_1_out_ap_vld;
output  [39:0] psum_vec_out;
output   psum_vec_out_ap_vld;

reg ap_idle;
reg vq_buf_0_ce0;
reg vq_buf_1_ce0;
reg vq_buf_2_ce0;
reg vq_buf_3_ce0;
reg vq_buf_4_ce0;
reg vq_buf_5_ce0;
reg vq_buf_6_ce0;
reg vq_buf_7_ce0;
reg vs_buf_ce0;
reg rq_buf_ce0;
reg rq_buf_1_ce0;
reg rq_buf_2_ce0;
reg rq_buf_3_ce0;
reg rq_buf_4_ce0;
reg rq_buf_5_ce0;
reg rq_buf_6_ce0;
reg rq_buf_7_ce0;
reg rs_buf_ce0;
reg psum_vec_7_out_ap_vld;
reg psum_vec_6_out_ap_vld;
reg psum_vec_5_out_ap_vld;
reg psum_vec_4_out_ap_vld;
reg psum_vec_3_out_ap_vld;
reg psum_vec_2_out_ap_vld;
reg psum_vec_1_out_ap_vld;
reg psum_vec_out_ap_vld;

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
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter5_fsm_state6;
wire    ap_CS_iter4_fsm_state5;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter6_fsm_state7;
wire    ap_CS_iter7_fsm_state8;
wire    ap_CS_iter8_fsm_state9;
wire   [0:0] icmp_ln336_fu_678_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg   [7:0] st_1_reg_2897;
reg   [0:0] icmp_ln336_reg_2902;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter1_reg;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter2_reg;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter3_reg;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter4_reg;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter5_reg;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter6_reg;
reg   [0:0] icmp_ln336_reg_2902_pp0_iter7_reg;
wire   [63:0] zext_ln350_2_fu_700_p1;
reg   [63:0] zext_ln350_2_reg_2906;
reg   [63:0] zext_ln350_2_reg_2906_pp0_iter1_reg;
reg   [63:0] zext_ln350_2_reg_2906_pp0_iter2_reg;
reg   [63:0] zext_ln350_2_reg_2906_pp0_iter3_reg;
wire   [63:0] zext_ln336_fu_717_p1;
reg   [63:0] zext_ln336_reg_2951;
reg   [63:0] zext_ln336_reg_2951_pp0_iter2_reg;
reg   [63:0] zext_ln336_reg_2951_pp0_iter3_reg;
reg   [63:0] zext_ln336_reg_2951_pp0_iter4_reg;
wire  signed [15:0] sext_ln350_fu_732_p1;
reg   [7:0] rq_buf_1_load_reg_3013;
wire   [7:0] trunc_ln350_1_fu_740_p1;
reg  signed [7:0] trunc_ln350_1_reg_3018;
reg   [7:0] rq_buf_2_load_reg_3023;
wire   [7:0] trunc_ln350_2_fu_744_p1;
reg  signed [7:0] trunc_ln350_2_reg_3028;
wire  signed [15:0] sext_ln350_9_fu_752_p1;
reg   [7:0] rq_buf_4_load_reg_3050;
wire   [7:0] trunc_ln350_4_fu_760_p1;
reg  signed [7:0] trunc_ln350_4_reg_3055;
wire  signed [15:0] sext_ln350_15_fu_768_p1;
reg   [7:0] rq_buf_6_load_reg_3077;
wire   [7:0] trunc_ln350_6_fu_776_p1;
reg  signed [7:0] trunc_ln350_6_reg_3082;
wire  signed [15:0] sext_ln350_21_fu_784_p1;
reg  signed [7:0] tmp_1_reg_3109;
reg  signed [7:0] tmp_2_reg_3114;
reg  signed [7:0] tmp_4_reg_3124;
reg  signed [7:0] tmp_6_reg_3134;
reg  signed [7:0] tmp_9_reg_3149;
reg  signed [7:0] tmp_10_reg_3154;
reg  signed [7:0] tmp_12_reg_3164;
reg  signed [7:0] tmp_14_reg_3174;
reg  signed [7:0] tmp_17_reg_3189;
reg  signed [7:0] tmp_18_reg_3194;
reg  signed [7:0] tmp_20_reg_3204;
reg  signed [7:0] tmp_22_reg_3214;
reg  signed [7:0] tmp_25_reg_3229;
reg  signed [7:0] tmp_26_reg_3234;
reg  signed [7:0] tmp_28_reg_3244;
reg  signed [7:0] tmp_30_reg_3254;
reg  signed [7:0] tmp_33_reg_3269;
reg  signed [7:0] tmp_34_reg_3274;
reg  signed [7:0] tmp_36_reg_3284;
reg  signed [7:0] tmp_38_reg_3294;
reg  signed [7:0] tmp_41_reg_3309;
reg  signed [7:0] tmp_42_reg_3314;
reg  signed [7:0] tmp_44_reg_3324;
reg  signed [7:0] tmp_46_reg_3334;
reg  signed [7:0] tmp_49_reg_3349;
reg  signed [7:0] tmp_50_reg_3354;
reg  signed [7:0] tmp_52_reg_3364;
reg  signed [7:0] tmp_54_reg_3374;
wire  signed [15:0] mul_res_1_fu_1470_p2;
reg  signed [15:0] mul_res_1_reg_3384;
wire  signed [15:0] mul_res_2_fu_1482_p2;
reg  signed [15:0] mul_res_2_reg_3389;
wire  signed [15:0] mul_res_4_fu_1494_p2;
reg  signed [15:0] mul_res_4_reg_3394;
wire  signed [15:0] mul_res_6_fu_1506_p2;
reg  signed [15:0] mul_res_6_reg_3399;
wire  signed [15:0] mul_res_9_fu_1515_p2;
reg  signed [15:0] mul_res_9_reg_3404;
wire  signed [15:0] mul_res_10_fu_1524_p2;
reg  signed [15:0] mul_res_10_reg_3409;
wire  signed [15:0] mul_res_12_fu_1533_p2;
reg  signed [15:0] mul_res_12_reg_3414;
wire  signed [15:0] mul_res_14_fu_1542_p2;
reg  signed [15:0] mul_res_14_reg_3419;
wire  signed [15:0] mul_res_17_fu_1551_p2;
reg  signed [15:0] mul_res_17_reg_3424;
wire  signed [15:0] mul_res_18_fu_1560_p2;
reg  signed [15:0] mul_res_18_reg_3429;
wire  signed [15:0] mul_res_20_fu_1569_p2;
reg  signed [15:0] mul_res_20_reg_3434;
wire  signed [15:0] mul_res_22_fu_1578_p2;
reg  signed [15:0] mul_res_22_reg_3439;
wire  signed [15:0] mul_res_25_fu_1587_p2;
reg  signed [15:0] mul_res_25_reg_3444;
wire  signed [15:0] mul_res_26_fu_1596_p2;
reg  signed [15:0] mul_res_26_reg_3449;
wire  signed [15:0] mul_res_28_fu_1605_p2;
reg  signed [15:0] mul_res_28_reg_3454;
wire  signed [15:0] mul_res_30_fu_1614_p2;
reg  signed [15:0] mul_res_30_reg_3459;
wire  signed [15:0] mul_res_33_fu_1623_p2;
reg  signed [15:0] mul_res_33_reg_3464;
wire  signed [15:0] mul_res_34_fu_1632_p2;
reg  signed [15:0] mul_res_34_reg_3469;
wire  signed [15:0] mul_res_36_fu_1641_p2;
reg  signed [15:0] mul_res_36_reg_3474;
wire  signed [15:0] mul_res_38_fu_1650_p2;
reg  signed [15:0] mul_res_38_reg_3479;
wire  signed [15:0] mul_res_41_fu_1659_p2;
reg  signed [15:0] mul_res_41_reg_3484;
wire  signed [15:0] mul_res_42_fu_1668_p2;
reg  signed [15:0] mul_res_42_reg_3489;
wire  signed [15:0] mul_res_44_fu_1677_p2;
reg  signed [15:0] mul_res_44_reg_3494;
wire  signed [15:0] mul_res_46_fu_1686_p2;
reg  signed [15:0] mul_res_46_reg_3499;
wire  signed [15:0] mul_res_49_fu_1695_p2;
reg  signed [15:0] mul_res_49_reg_3504;
wire  signed [15:0] mul_res_50_fu_1704_p2;
reg  signed [15:0] mul_res_50_reg_3509;
wire  signed [15:0] mul_res_52_fu_1713_p2;
reg  signed [15:0] mul_res_52_reg_3514;
wire  signed [15:0] mul_res_54_fu_1722_p2;
reg  signed [15:0] mul_res_54_reg_3519;
wire  signed [15:0] mul_res_57_fu_1731_p2;
reg  signed [15:0] mul_res_57_reg_3524;
wire  signed [15:0] mul_res_58_fu_1740_p2;
reg  signed [15:0] mul_res_58_reg_3529;
wire  signed [15:0] mul_res_60_fu_1749_p2;
reg  signed [15:0] mul_res_60_reg_3534;
wire  signed [15:0] mul_res_62_fu_1758_p2;
reg  signed [15:0] mul_res_62_reg_3539;
wire   [17:0] add_ln352_2_fu_1866_p2;
reg   [17:0] add_ln352_2_reg_3709;
wire   [17:0] add_ln352_5_fu_1878_p2;
reg   [17:0] add_ln352_5_reg_3714;
wire   [17:0] add_ln352_9_fu_1890_p2;
reg   [17:0] add_ln352_9_reg_3719;
wire   [17:0] add_ln352_12_fu_1902_p2;
reg   [17:0] add_ln352_12_reg_3724;
wire   [17:0] add_ln352_16_fu_1914_p2;
reg   [17:0] add_ln352_16_reg_3729;
wire   [17:0] add_ln352_19_fu_1926_p2;
reg   [17:0] add_ln352_19_reg_3734;
wire   [17:0] add_ln352_23_fu_1938_p2;
reg   [17:0] add_ln352_23_reg_3739;
wire   [17:0] add_ln352_26_fu_1950_p2;
reg   [17:0] add_ln352_26_reg_3744;
wire   [17:0] add_ln352_30_fu_1962_p2;
reg   [17:0] add_ln352_30_reg_3749;
wire   [17:0] add_ln352_33_fu_1974_p2;
reg   [17:0] add_ln352_33_reg_3754;
wire   [17:0] add_ln352_37_fu_1986_p2;
reg   [17:0] add_ln352_37_reg_3759;
wire   [17:0] add_ln352_40_fu_1998_p2;
reg   [17:0] add_ln352_40_reg_3764;
wire   [17:0] add_ln352_44_fu_2010_p2;
reg   [17:0] add_ln352_44_reg_3769;
wire   [17:0] add_ln352_47_fu_2022_p2;
reg   [17:0] add_ln352_47_reg_3774;
wire   [17:0] add_ln352_51_fu_2034_p2;
reg   [17:0] add_ln352_51_reg_3779;
wire   [17:0] add_ln352_54_fu_2046_p2;
reg   [17:0] add_ln352_54_reg_3784;
wire   [18:0] add_ln352_6_fu_2058_p2;
reg   [18:0] add_ln352_6_reg_3794;
wire   [18:0] add_ln352_13_fu_2070_p2;
reg   [18:0] add_ln352_13_reg_3799;
wire   [18:0] add_ln352_20_fu_2082_p2;
reg   [18:0] add_ln352_20_reg_3804;
wire   [18:0] add_ln352_27_fu_2094_p2;
reg   [18:0] add_ln352_27_reg_3809;
wire   [18:0] add_ln352_34_fu_2106_p2;
reg   [18:0] add_ln352_34_reg_3814;
wire   [18:0] add_ln352_41_fu_2118_p2;
reg   [18:0] add_ln352_41_reg_3819;
wire   [18:0] add_ln352_48_fu_2130_p2;
reg   [18:0] add_ln352_48_reg_3824;
wire   [18:0] add_ln352_55_fu_2142_p2;
reg   [18:0] add_ln352_55_reg_3829;
wire   [4:0] add_ln360_fu_2160_p2;
reg   [4:0] add_ln360_reg_3834;
wire   [4:0] add_ln360_2_fu_2180_p2;
reg   [4:0] add_ln360_2_reg_3839;
wire   [4:0] add_ln360_4_fu_2200_p2;
reg   [4:0] add_ln360_4_reg_3844;
wire   [4:0] add_ln360_6_fu_2220_p2;
reg   [4:0] add_ln360_6_reg_3849;
wire   [4:0] add_ln360_8_fu_2240_p2;
reg   [4:0] add_ln360_8_reg_3854;
wire   [4:0] add_ln360_10_fu_2260_p2;
reg   [4:0] add_ln360_10_reg_3859;
wire   [4:0] add_ln360_12_fu_2280_p2;
reg   [4:0] add_ln360_12_reg_3864;
wire   [4:0] add_ln360_14_fu_2300_p2;
reg   [4:0] add_ln360_14_reg_3869;
wire   [39:0] shl_ln360_fu_2333_p2;
reg   [39:0] shl_ln360_reg_3874;
wire   [39:0] shl_ln360_1_fu_2342_p2;
reg   [39:0] shl_ln360_1_reg_3879;
wire   [39:0] shl_ln360_2_fu_2351_p2;
reg   [39:0] shl_ln360_2_reg_3884;
wire   [39:0] shl_ln360_3_fu_2360_p2;
reg   [39:0] shl_ln360_3_reg_3889;
wire   [39:0] shl_ln360_4_fu_2369_p2;
reg   [39:0] shl_ln360_4_reg_3894;
wire   [39:0] shl_ln360_5_fu_2378_p2;
reg   [39:0] shl_ln360_5_reg_3899;
wire   [39:0] shl_ln360_6_fu_2387_p2;
reg   [39:0] shl_ln360_6_reg_3904;
wire   [39:0] shl_ln360_7_fu_2396_p2;
reg   [39:0] shl_ln360_7_reg_3909;
reg   [39:0] psum_vec_fu_292;
wire   [39:0] add_ln360_1_fu_2426_p2;
wire    ap_loop_init;
reg   [39:0] psum_vec_1_fu_296;
wire   [39:0] add_ln360_3_fu_2431_p2;
reg   [39:0] psum_vec_2_fu_300;
wire   [39:0] add_ln360_5_fu_2436_p2;
reg   [39:0] psum_vec_3_fu_304;
wire   [39:0] add_ln360_7_fu_2441_p2;
reg   [39:0] psum_vec_4_fu_308;
wire   [39:0] add_ln360_9_fu_2446_p2;
reg   [39:0] psum_vec_5_fu_312;
wire   [39:0] add_ln360_11_fu_2451_p2;
reg   [39:0] psum_vec_6_fu_316;
wire   [39:0] add_ln360_13_fu_2456_p2;
reg   [39:0] psum_vec_7_fu_320;
wire   [39:0] add_ln360_15_fu_2461_p2;
reg   [7:0] st_fu_324;
wire   [7:0] add_ln336_fu_684_p2;
reg   [7:0] ap_sig_allocacmp_st_1;
wire   [9:0] zext_ln350_1_fu_690_p1;
wire   [9:0] add_ln350_fu_694_p2;
wire  signed [7:0] trunc_ln350_fu_728_p1;
wire  signed [7:0] trunc_ln350_3_fu_748_p1;
wire  signed [7:0] trunc_ln350_5_fu_764_p1;
wire  signed [7:0] trunc_ln350_7_fu_780_p1;
wire  signed [7:0] tmp_s_fu_792_p4;
wire  signed [7:0] tmp_3_fu_826_p4;
wire  signed [7:0] tmp_5_fu_850_p4;
wire  signed [7:0] tmp_7_fu_874_p4;
wire  signed [7:0] tmp_8_fu_888_p4;
wire  signed [7:0] tmp_11_fu_922_p4;
wire  signed [7:0] tmp_13_fu_946_p4;
wire  signed [7:0] tmp_15_fu_970_p4;
wire  signed [7:0] tmp_16_fu_984_p4;
wire  signed [7:0] tmp_19_fu_1018_p4;
wire  signed [7:0] tmp_21_fu_1042_p4;
wire  signed [7:0] tmp_23_fu_1066_p4;
wire  signed [7:0] tmp_24_fu_1080_p4;
wire  signed [7:0] tmp_27_fu_1114_p4;
wire  signed [7:0] tmp_29_fu_1138_p4;
wire  signed [7:0] tmp_31_fu_1162_p4;
wire  signed [7:0] tmp_32_fu_1176_p4;
wire  signed [7:0] tmp_35_fu_1210_p4;
wire  signed [7:0] tmp_37_fu_1234_p4;
wire  signed [7:0] tmp_39_fu_1258_p4;
wire  signed [7:0] tmp_40_fu_1272_p4;
wire  signed [7:0] tmp_43_fu_1306_p4;
wire  signed [7:0] tmp_45_fu_1330_p4;
wire  signed [7:0] tmp_47_fu_1354_p4;
wire  signed [7:0] tmp_48_fu_1368_p4;
wire  signed [7:0] tmp_51_fu_1402_p4;
wire  signed [7:0] tmp_53_fu_1426_p4;
wire  signed [7:0] tmp_55_fu_1450_p4;
wire  signed [7:0] mul_res_1_fu_1470_p1;
wire  signed [15:0] sext_ln350_3_fu_1464_p1;
wire  signed [7:0] mul_res_2_fu_1482_p1;
wire  signed [15:0] sext_ln350_6_fu_1476_p1;
wire  signed [7:0] mul_res_4_fu_1494_p1;
wire  signed [15:0] sext_ln350_12_fu_1488_p1;
wire  signed [7:0] mul_res_6_fu_1506_p1;
wire  signed [15:0] sext_ln350_18_fu_1500_p1;
wire  signed [7:0] mul_res_9_fu_1515_p1;
wire  signed [7:0] mul_res_10_fu_1524_p1;
wire  signed [7:0] mul_res_12_fu_1533_p1;
wire  signed [7:0] mul_res_14_fu_1542_p1;
wire  signed [7:0] mul_res_17_fu_1551_p1;
wire  signed [7:0] mul_res_18_fu_1560_p1;
wire  signed [7:0] mul_res_20_fu_1569_p1;
wire  signed [7:0] mul_res_22_fu_1578_p1;
wire  signed [7:0] mul_res_25_fu_1587_p1;
wire  signed [7:0] mul_res_26_fu_1596_p1;
wire  signed [7:0] mul_res_28_fu_1605_p1;
wire  signed [7:0] mul_res_30_fu_1614_p1;
wire  signed [7:0] mul_res_33_fu_1623_p1;
wire  signed [7:0] mul_res_34_fu_1632_p1;
wire  signed [7:0] mul_res_36_fu_1641_p1;
wire  signed [7:0] mul_res_38_fu_1650_p1;
wire  signed [7:0] mul_res_41_fu_1659_p1;
wire  signed [7:0] mul_res_42_fu_1668_p1;
wire  signed [7:0] mul_res_44_fu_1677_p1;
wire  signed [7:0] mul_res_46_fu_1686_p1;
wire  signed [7:0] mul_res_49_fu_1695_p1;
wire  signed [7:0] mul_res_50_fu_1704_p1;
wire  signed [7:0] mul_res_52_fu_1713_p1;
wire  signed [7:0] mul_res_54_fu_1722_p1;
wire  signed [7:0] mul_res_57_fu_1731_p1;
wire  signed [7:0] mul_res_58_fu_1740_p1;
wire  signed [7:0] mul_res_60_fu_1749_p1;
wire  signed [7:0] mul_res_62_fu_1758_p1;
wire  signed [16:0] grp_fu_2538_p3;
wire  signed [16:0] grp_fu_2547_p3;
wire  signed [17:0] sext_ln352_1_fu_1863_p1;
wire  signed [17:0] sext_ln352_fu_1860_p1;
wire  signed [16:0] grp_fu_2556_p3;
wire  signed [16:0] grp_fu_2565_p3;
wire  signed [17:0] sext_ln352_4_fu_1875_p1;
wire  signed [17:0] sext_ln352_3_fu_1872_p1;
wire  signed [16:0] grp_fu_2574_p3;
wire  signed [16:0] grp_fu_2583_p3;
wire  signed [17:0] sext_ln352_7_fu_1887_p1;
wire  signed [17:0] sext_ln352_6_fu_1884_p1;
wire  signed [16:0] grp_fu_2592_p3;
wire  signed [16:0] grp_fu_2601_p3;
wire  signed [17:0] sext_ln352_10_fu_1899_p1;
wire  signed [17:0] sext_ln352_9_fu_1896_p1;
wire  signed [16:0] grp_fu_2610_p3;
wire  signed [16:0] grp_fu_2619_p3;
wire  signed [17:0] sext_ln352_13_fu_1911_p1;
wire  signed [17:0] sext_ln352_12_fu_1908_p1;
wire  signed [16:0] grp_fu_2628_p3;
wire  signed [16:0] grp_fu_2637_p3;
wire  signed [17:0] sext_ln352_16_fu_1923_p1;
wire  signed [17:0] sext_ln352_15_fu_1920_p1;
wire  signed [16:0] grp_fu_2646_p3;
wire  signed [16:0] grp_fu_2655_p3;
wire  signed [17:0] sext_ln352_19_fu_1935_p1;
wire  signed [17:0] sext_ln352_18_fu_1932_p1;
wire  signed [16:0] grp_fu_2664_p3;
wire  signed [16:0] grp_fu_2673_p3;
wire  signed [17:0] sext_ln352_22_fu_1947_p1;
wire  signed [17:0] sext_ln352_21_fu_1944_p1;
wire  signed [16:0] grp_fu_2682_p3;
wire  signed [16:0] grp_fu_2691_p3;
wire  signed [17:0] sext_ln352_25_fu_1959_p1;
wire  signed [17:0] sext_ln352_24_fu_1956_p1;
wire  signed [16:0] grp_fu_2700_p3;
wire  signed [16:0] grp_fu_2709_p3;
wire  signed [17:0] sext_ln352_28_fu_1971_p1;
wire  signed [17:0] sext_ln352_27_fu_1968_p1;
wire  signed [16:0] grp_fu_2718_p3;
wire  signed [16:0] grp_fu_2727_p3;
wire  signed [17:0] sext_ln352_31_fu_1983_p1;
wire  signed [17:0] sext_ln352_30_fu_1980_p1;
wire  signed [16:0] grp_fu_2736_p3;
wire  signed [16:0] grp_fu_2745_p3;
wire  signed [17:0] sext_ln352_34_fu_1995_p1;
wire  signed [17:0] sext_ln352_33_fu_1992_p1;
wire  signed [16:0] grp_fu_2754_p3;
wire  signed [16:0] grp_fu_2763_p3;
wire  signed [17:0] sext_ln352_37_fu_2007_p1;
wire  signed [17:0] sext_ln352_36_fu_2004_p1;
wire  signed [16:0] grp_fu_2772_p3;
wire  signed [16:0] grp_fu_2781_p3;
wire  signed [17:0] sext_ln352_40_fu_2019_p1;
wire  signed [17:0] sext_ln352_39_fu_2016_p1;
wire  signed [16:0] grp_fu_2790_p3;
wire  signed [16:0] grp_fu_2799_p3;
wire  signed [17:0] sext_ln352_43_fu_2031_p1;
wire  signed [17:0] sext_ln352_42_fu_2028_p1;
wire  signed [16:0] grp_fu_2808_p3;
wire  signed [16:0] grp_fu_2817_p3;
wire  signed [17:0] sext_ln352_46_fu_2043_p1;
wire  signed [17:0] sext_ln352_45_fu_2040_p1;
wire  signed [18:0] sext_ln352_5_fu_2055_p1;
wire  signed [18:0] sext_ln352_2_fu_2052_p1;
wire  signed [18:0] sext_ln352_11_fu_2067_p1;
wire  signed [18:0] sext_ln352_8_fu_2064_p1;
wire  signed [18:0] sext_ln352_17_fu_2079_p1;
wire  signed [18:0] sext_ln352_14_fu_2076_p1;
wire  signed [18:0] sext_ln352_23_fu_2091_p1;
wire  signed [18:0] sext_ln352_20_fu_2088_p1;
wire  signed [18:0] sext_ln352_29_fu_2103_p1;
wire  signed [18:0] sext_ln352_26_fu_2100_p1;
wire  signed [18:0] sext_ln352_35_fu_2115_p1;
wire  signed [18:0] sext_ln352_32_fu_2112_p1;
wire  signed [18:0] sext_ln352_41_fu_2127_p1;
wire  signed [18:0] sext_ln352_38_fu_2124_p1;
wire  signed [18:0] sext_ln352_47_fu_2139_p1;
wire  signed [18:0] sext_ln352_44_fu_2136_p1;
wire   [3:0] v_shift_fu_2148_p1;
wire   [4:0] zext_ln360_1_fu_2156_p1;
wire   [4:0] zext_ln360_fu_2152_p1;
wire   [3:0] v_shift_1_fu_2166_p4;
wire   [4:0] zext_ln360_3_fu_2176_p1;
wire   [3:0] v_shift_2_fu_2186_p4;
wire   [4:0] zext_ln360_5_fu_2196_p1;
wire   [3:0] v_shift_3_fu_2206_p4;
wire   [4:0] zext_ln360_7_fu_2216_p1;
wire   [3:0] v_shift_4_fu_2226_p4;
wire   [4:0] zext_ln360_9_fu_2236_p1;
wire   [3:0] v_shift_5_fu_2246_p4;
wire   [4:0] zext_ln360_11_fu_2256_p1;
wire   [3:0] v_shift_6_fu_2266_p4;
wire   [4:0] zext_ln360_13_fu_2276_p1;
wire   [3:0] v_shift_7_fu_2286_p4;
wire   [4:0] zext_ln360_15_fu_2296_p1;
wire  signed [39:0] sext_ln350_24_fu_2306_p1;
wire   [39:0] zext_ln360_2_fu_2330_p1;
wire  signed [39:0] sext_ln350_41_fu_2309_p1;
wire   [39:0] zext_ln360_4_fu_2339_p1;
wire  signed [39:0] sext_ln350_58_fu_2312_p1;
wire   [39:0] zext_ln360_6_fu_2348_p1;
wire  signed [39:0] sext_ln350_75_fu_2315_p1;
wire   [39:0] zext_ln360_8_fu_2357_p1;
wire  signed [39:0] sext_ln350_92_fu_2318_p1;
wire   [39:0] zext_ln360_10_fu_2366_p1;
wire  signed [39:0] sext_ln350_109_fu_2321_p1;
wire   [39:0] zext_ln360_12_fu_2375_p1;
wire  signed [39:0] sext_ln350_126_fu_2324_p1;
wire   [39:0] zext_ln360_14_fu_2384_p1;
wire  signed [39:0] sext_ln356_fu_2327_p1;
wire   [39:0] zext_ln360_16_fu_2393_p1;
wire  signed [7:0] grp_fu_2538_p1;
wire  signed [7:0] grp_fu_2547_p1;
wire  signed [7:0] grp_fu_2556_p1;
wire  signed [7:0] grp_fu_2565_p1;
wire  signed [7:0] grp_fu_2574_p1;
wire  signed [7:0] grp_fu_2583_p1;
wire  signed [7:0] grp_fu_2592_p1;
wire  signed [7:0] grp_fu_2601_p1;
wire  signed [7:0] grp_fu_2610_p1;
wire  signed [7:0] grp_fu_2619_p1;
wire  signed [7:0] grp_fu_2628_p1;
wire  signed [7:0] grp_fu_2637_p1;
wire  signed [7:0] grp_fu_2646_p1;
wire  signed [7:0] grp_fu_2655_p1;
wire  signed [7:0] grp_fu_2664_p1;
wire  signed [7:0] grp_fu_2673_p1;
wire  signed [7:0] grp_fu_2682_p1;
wire  signed [7:0] grp_fu_2691_p1;
wire  signed [7:0] grp_fu_2700_p1;
wire  signed [7:0] grp_fu_2709_p1;
wire  signed [7:0] grp_fu_2718_p1;
wire  signed [7:0] grp_fu_2727_p1;
wire  signed [7:0] grp_fu_2736_p1;
wire  signed [7:0] grp_fu_2745_p1;
wire  signed [7:0] grp_fu_2754_p1;
wire  signed [7:0] grp_fu_2763_p1;
wire  signed [7:0] grp_fu_2772_p1;
wire  signed [7:0] grp_fu_2781_p1;
wire  signed [7:0] grp_fu_2790_p1;
wire  signed [7:0] grp_fu_2799_p1;
wire  signed [7:0] grp_fu_2808_p1;
wire  signed [7:0] grp_fu_2817_p1;
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
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg   [1:0] ap_NS_iter6_fsm;
reg   [1:0] ap_NS_iter7_fsm;
reg   [1:0] ap_NS_iter8_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
wire    ap_ST_iter5_fsm_state6_blk;
wire    ap_ST_iter6_fsm_state7_blk;
wire    ap_ST_iter7_fsm_state8_blk;
wire    ap_ST_iter8_fsm_state9_blk;
wire    ap_start_int;
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
//#0 psum_vec_fu_292 = 40'd0;
//#0 psum_vec_1_fu_296 = 40'd0;
//#0 psum_vec_2_fu_300 = 40'd0;
//#0 psum_vec_3_fu_304 = 40'd0;
//#0 psum_vec_4_fu_308 = 40'd0;
//#0 psum_vec_5_fu_312 = 40'd0;
//#0 psum_vec_6_fu_316 = 40'd0;
//#0 psum_vec_7_fu_320 = 40'd0;
//#0 st_fu_324 = 8'd0;
//#0 ap_done_reg = 1'b0;
end

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U194(
    .din0(trunc_ln350_1_reg_3018),
    .din1(mul_res_1_fu_1470_p1),
    .dout(mul_res_1_fu_1470_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U195(
    .din0(trunc_ln350_2_reg_3028),
    .din1(mul_res_2_fu_1482_p1),
    .dout(mul_res_2_fu_1482_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U196(
    .din0(trunc_ln350_4_reg_3055),
    .din1(mul_res_4_fu_1494_p1),
    .dout(mul_res_4_fu_1494_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U197(
    .din0(trunc_ln350_6_reg_3082),
    .din1(mul_res_6_fu_1506_p1),
    .dout(mul_res_6_fu_1506_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U198(
    .din0(tmp_1_reg_3109),
    .din1(mul_res_9_fu_1515_p1),
    .dout(mul_res_9_fu_1515_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U199(
    .din0(tmp_2_reg_3114),
    .din1(mul_res_10_fu_1524_p1),
    .dout(mul_res_10_fu_1524_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U200(
    .din0(tmp_4_reg_3124),
    .din1(mul_res_12_fu_1533_p1),
    .dout(mul_res_12_fu_1533_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U201(
    .din0(tmp_6_reg_3134),
    .din1(mul_res_14_fu_1542_p1),
    .dout(mul_res_14_fu_1542_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U202(
    .din0(tmp_9_reg_3149),
    .din1(mul_res_17_fu_1551_p1),
    .dout(mul_res_17_fu_1551_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U203(
    .din0(tmp_10_reg_3154),
    .din1(mul_res_18_fu_1560_p1),
    .dout(mul_res_18_fu_1560_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U204(
    .din0(tmp_12_reg_3164),
    .din1(mul_res_20_fu_1569_p1),
    .dout(mul_res_20_fu_1569_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U205(
    .din0(tmp_14_reg_3174),
    .din1(mul_res_22_fu_1578_p1),
    .dout(mul_res_22_fu_1578_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U206(
    .din0(tmp_17_reg_3189),
    .din1(mul_res_25_fu_1587_p1),
    .dout(mul_res_25_fu_1587_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U207(
    .din0(tmp_18_reg_3194),
    .din1(mul_res_26_fu_1596_p1),
    .dout(mul_res_26_fu_1596_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U208(
    .din0(tmp_20_reg_3204),
    .din1(mul_res_28_fu_1605_p1),
    .dout(mul_res_28_fu_1605_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U209(
    .din0(tmp_22_reg_3214),
    .din1(mul_res_30_fu_1614_p1),
    .dout(mul_res_30_fu_1614_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U210(
    .din0(tmp_25_reg_3229),
    .din1(mul_res_33_fu_1623_p1),
    .dout(mul_res_33_fu_1623_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U211(
    .din0(tmp_26_reg_3234),
    .din1(mul_res_34_fu_1632_p1),
    .dout(mul_res_34_fu_1632_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U212(
    .din0(tmp_28_reg_3244),
    .din1(mul_res_36_fu_1641_p1),
    .dout(mul_res_36_fu_1641_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U213(
    .din0(tmp_30_reg_3254),
    .din1(mul_res_38_fu_1650_p1),
    .dout(mul_res_38_fu_1650_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U214(
    .din0(tmp_33_reg_3269),
    .din1(mul_res_41_fu_1659_p1),
    .dout(mul_res_41_fu_1659_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U215(
    .din0(tmp_34_reg_3274),
    .din1(mul_res_42_fu_1668_p1),
    .dout(mul_res_42_fu_1668_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U216(
    .din0(tmp_36_reg_3284),
    .din1(mul_res_44_fu_1677_p1),
    .dout(mul_res_44_fu_1677_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U217(
    .din0(tmp_38_reg_3294),
    .din1(mul_res_46_fu_1686_p1),
    .dout(mul_res_46_fu_1686_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U218(
    .din0(tmp_41_reg_3309),
    .din1(mul_res_49_fu_1695_p1),
    .dout(mul_res_49_fu_1695_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U219(
    .din0(tmp_42_reg_3314),
    .din1(mul_res_50_fu_1704_p1),
    .dout(mul_res_50_fu_1704_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U220(
    .din0(tmp_44_reg_3324),
    .din1(mul_res_52_fu_1713_p1),
    .dout(mul_res_52_fu_1713_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U221(
    .din0(tmp_46_reg_3334),
    .din1(mul_res_54_fu_1722_p1),
    .dout(mul_res_54_fu_1722_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U222(
    .din0(tmp_49_reg_3349),
    .din1(mul_res_57_fu_1731_p1),
    .dout(mul_res_57_fu_1731_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U223(
    .din0(tmp_50_reg_3354),
    .din1(mul_res_58_fu_1740_p1),
    .dout(mul_res_58_fu_1740_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U224(
    .din0(tmp_52_reg_3364),
    .din1(mul_res_60_fu_1749_p1),
    .dout(mul_res_60_fu_1749_p2)
);

RV_GEMM_mul_8s_8s_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_8s_8s_16_1_1_U225(
    .din0(tmp_54_reg_3374),
    .din1(mul_res_62_fu_1758_p1),
    .dout(mul_res_62_fu_1758_p2)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U226(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln350_fu_728_p1),
    .din1(grp_fu_2538_p1),
    .din2(mul_res_1_reg_3384),
    .ce(1'b1),
    .dout(grp_fu_2538_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U227(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln350_3_fu_748_p1),
    .din1(grp_fu_2547_p1),
    .din2(mul_res_2_reg_3389),
    .ce(1'b1),
    .dout(grp_fu_2547_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U228(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln350_5_fu_764_p1),
    .din1(grp_fu_2556_p1),
    .din2(mul_res_4_reg_3394),
    .ce(1'b1),
    .dout(grp_fu_2556_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U229(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln350_7_fu_780_p1),
    .din1(grp_fu_2565_p1),
    .din2(mul_res_6_reg_3399),
    .ce(1'b1),
    .dout(grp_fu_2565_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U230(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_s_fu_792_p4),
    .din1(grp_fu_2574_p1),
    .din2(mul_res_9_reg_3404),
    .ce(1'b1),
    .dout(grp_fu_2574_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U231(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_3_fu_826_p4),
    .din1(grp_fu_2583_p1),
    .din2(mul_res_10_reg_3409),
    .ce(1'b1),
    .dout(grp_fu_2583_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U232(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_5_fu_850_p4),
    .din1(grp_fu_2592_p1),
    .din2(mul_res_12_reg_3414),
    .ce(1'b1),
    .dout(grp_fu_2592_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U233(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_7_fu_874_p4),
    .din1(grp_fu_2601_p1),
    .din2(mul_res_14_reg_3419),
    .ce(1'b1),
    .dout(grp_fu_2601_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U234(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_8_fu_888_p4),
    .din1(grp_fu_2610_p1),
    .din2(mul_res_17_reg_3424),
    .ce(1'b1),
    .dout(grp_fu_2610_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U235(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_11_fu_922_p4),
    .din1(grp_fu_2619_p1),
    .din2(mul_res_18_reg_3429),
    .ce(1'b1),
    .dout(grp_fu_2619_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U236(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_13_fu_946_p4),
    .din1(grp_fu_2628_p1),
    .din2(mul_res_20_reg_3434),
    .ce(1'b1),
    .dout(grp_fu_2628_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U237(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_15_fu_970_p4),
    .din1(grp_fu_2637_p1),
    .din2(mul_res_22_reg_3439),
    .ce(1'b1),
    .dout(grp_fu_2637_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U238(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_16_fu_984_p4),
    .din1(grp_fu_2646_p1),
    .din2(mul_res_25_reg_3444),
    .ce(1'b1),
    .dout(grp_fu_2646_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U239(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_19_fu_1018_p4),
    .din1(grp_fu_2655_p1),
    .din2(mul_res_26_reg_3449),
    .ce(1'b1),
    .dout(grp_fu_2655_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U240(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_21_fu_1042_p4),
    .din1(grp_fu_2664_p1),
    .din2(mul_res_28_reg_3454),
    .ce(1'b1),
    .dout(grp_fu_2664_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U241(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_23_fu_1066_p4),
    .din1(grp_fu_2673_p1),
    .din2(mul_res_30_reg_3459),
    .ce(1'b1),
    .dout(grp_fu_2673_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U242(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_24_fu_1080_p4),
    .din1(grp_fu_2682_p1),
    .din2(mul_res_33_reg_3464),
    .ce(1'b1),
    .dout(grp_fu_2682_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U243(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_27_fu_1114_p4),
    .din1(grp_fu_2691_p1),
    .din2(mul_res_34_reg_3469),
    .ce(1'b1),
    .dout(grp_fu_2691_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U244(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_29_fu_1138_p4),
    .din1(grp_fu_2700_p1),
    .din2(mul_res_36_reg_3474),
    .ce(1'b1),
    .dout(grp_fu_2700_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U245(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_31_fu_1162_p4),
    .din1(grp_fu_2709_p1),
    .din2(mul_res_38_reg_3479),
    .ce(1'b1),
    .dout(grp_fu_2709_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U246(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_32_fu_1176_p4),
    .din1(grp_fu_2718_p1),
    .din2(mul_res_41_reg_3484),
    .ce(1'b1),
    .dout(grp_fu_2718_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U247(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_35_fu_1210_p4),
    .din1(grp_fu_2727_p1),
    .din2(mul_res_42_reg_3489),
    .ce(1'b1),
    .dout(grp_fu_2727_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U248(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_37_fu_1234_p4),
    .din1(grp_fu_2736_p1),
    .din2(mul_res_44_reg_3494),
    .ce(1'b1),
    .dout(grp_fu_2736_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U249(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_39_fu_1258_p4),
    .din1(grp_fu_2745_p1),
    .din2(mul_res_46_reg_3499),
    .ce(1'b1),
    .dout(grp_fu_2745_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U250(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_40_fu_1272_p4),
    .din1(grp_fu_2754_p1),
    .din2(mul_res_49_reg_3504),
    .ce(1'b1),
    .dout(grp_fu_2754_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U251(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_43_fu_1306_p4),
    .din1(grp_fu_2763_p1),
    .din2(mul_res_50_reg_3509),
    .ce(1'b1),
    .dout(grp_fu_2763_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U252(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_45_fu_1330_p4),
    .din1(grp_fu_2772_p1),
    .din2(mul_res_52_reg_3514),
    .ce(1'b1),
    .dout(grp_fu_2772_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U253(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_47_fu_1354_p4),
    .din1(grp_fu_2781_p1),
    .din2(mul_res_54_reg_3519),
    .ce(1'b1),
    .dout(grp_fu_2781_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U254(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_48_fu_1368_p4),
    .din1(grp_fu_2790_p1),
    .din2(mul_res_57_reg_3524),
    .ce(1'b1),
    .dout(grp_fu_2790_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U255(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_51_fu_1402_p4),
    .din1(grp_fu_2799_p1),
    .din2(mul_res_58_reg_3529),
    .ce(1'b1),
    .dout(grp_fu_2799_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U256(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_53_fu_1426_p4),
    .din1(grp_fu_2808_p1),
    .din2(mul_res_60_reg_3534),
    .ce(1'b1),
    .dout(grp_fu_2808_p3)
);

RV_GEMM_mac_muladd_8s_8s_16s_17_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 8 ),
    .din1_WIDTH( 8 ),
    .din2_WIDTH( 16 ),
    .dout_WIDTH( 17 ))
mac_muladd_8s_8s_16s_17_4_1_U257(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(tmp_55_fu_1450_p4),
    .din1(grp_fu_2817_p1),
    .din2(mul_res_62_reg_3539),
    .ce(1'b1),
    .dout(grp_fu_2817_p3)
);

RV_GEMM_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_CS_iter8_fsm <= ap_ST_iter8_fsm_state0;
    end else begin
        ap_CS_iter8_fsm <= ap_NS_iter8_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (ap_loop_exit_ready_pp0_iter8_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter8_reg <= 1'b0;
    end else if ((1'b1 == ap_CS_iter7_fsm_state8)) begin
        ap_loop_exit_ready_pp0_iter8_reg <= ap_loop_exit_ready_pp0_iter7_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_1_fu_296 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_1_fu_296 <= add_ln360_3_fu_2431_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_2_fu_300 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_2_fu_300 <= add_ln360_5_fu_2436_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_3_fu_304 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_3_fu_304 <= add_ln360_7_fu_2441_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_4_fu_308 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_4_fu_308 <= add_ln360_9_fu_2446_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_5_fu_312 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_5_fu_312 <= add_ln360_11_fu_2451_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_6_fu_316 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_6_fu_316 <= add_ln360_13_fu_2456_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_7_fu_320 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_7_fu_320 <= add_ln360_15_fu_2461_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (ap_loop_init == 1'b1))) begin
        psum_vec_fu_292 <= 40'd0;
    end else if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd0))) begin
        psum_vec_fu_292 <= add_ln360_1_fu_2426_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
        if ((icmp_ln336_fu_678_p2 == 1'd0)) begin
            st_fu_324 <= add_ln336_fu_684_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            st_fu_324 <= 8'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter5_fsm_state6)) begin
        add_ln352_12_reg_3724 <= add_ln352_12_fu_1902_p2;
        add_ln352_16_reg_3729 <= add_ln352_16_fu_1914_p2;
        add_ln352_19_reg_3734 <= add_ln352_19_fu_1926_p2;
        add_ln352_23_reg_3739 <= add_ln352_23_fu_1938_p2;
        add_ln352_26_reg_3744 <= add_ln352_26_fu_1950_p2;
        add_ln352_2_reg_3709 <= add_ln352_2_fu_1866_p2;
        add_ln352_30_reg_3749 <= add_ln352_30_fu_1962_p2;
        add_ln352_33_reg_3754 <= add_ln352_33_fu_1974_p2;
        add_ln352_37_reg_3759 <= add_ln352_37_fu_1986_p2;
        add_ln352_40_reg_3764 <= add_ln352_40_fu_1998_p2;
        add_ln352_44_reg_3769 <= add_ln352_44_fu_2010_p2;
        add_ln352_47_reg_3774 <= add_ln352_47_fu_2022_p2;
        add_ln352_51_reg_3779 <= add_ln352_51_fu_2034_p2;
        add_ln352_54_reg_3784 <= add_ln352_54_fu_2046_p2;
        add_ln352_5_reg_3714 <= add_ln352_5_fu_1878_p2;
        add_ln352_9_reg_3719 <= add_ln352_9_fu_1890_p2;
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        icmp_ln336_reg_2902_pp0_iter5_reg <= icmp_ln336_reg_2902_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter6_fsm_state7)) begin
        add_ln352_13_reg_3799 <= add_ln352_13_fu_2070_p2;
        add_ln352_20_reg_3804 <= add_ln352_20_fu_2082_p2;
        add_ln352_27_reg_3809 <= add_ln352_27_fu_2094_p2;
        add_ln352_34_reg_3814 <= add_ln352_34_fu_2106_p2;
        add_ln352_41_reg_3819 <= add_ln352_41_fu_2118_p2;
        add_ln352_48_reg_3824 <= add_ln352_48_fu_2130_p2;
        add_ln352_55_reg_3829 <= add_ln352_55_fu_2142_p2;
        add_ln352_6_reg_3794 <= add_ln352_6_fu_2058_p2;
        add_ln360_10_reg_3859 <= add_ln360_10_fu_2260_p2;
        add_ln360_12_reg_3864 <= add_ln360_12_fu_2280_p2;
        add_ln360_14_reg_3869 <= add_ln360_14_fu_2300_p2;
        add_ln360_2_reg_3839 <= add_ln360_2_fu_2180_p2;
        add_ln360_4_reg_3844 <= add_ln360_4_fu_2200_p2;
        add_ln360_6_reg_3849 <= add_ln360_6_fu_2220_p2;
        add_ln360_8_reg_3854 <= add_ln360_8_fu_2240_p2;
        add_ln360_reg_3834 <= add_ln360_fu_2160_p2;
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
        icmp_ln336_reg_2902_pp0_iter6_reg <= icmp_ln336_reg_2902_pp0_iter5_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln336_reg_2902 <= icmp_ln336_fu_678_p2;
        st_1_reg_2897 <= ap_sig_allocacmp_st_1;
        zext_ln350_2_reg_2906[9 : 0] <= zext_ln350_2_fu_700_p1[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln336_reg_2902_pp0_iter1_reg <= icmp_ln336_reg_2902;
        zext_ln336_reg_2951[7 : 0] <= zext_ln336_fu_717_p1[7 : 0];
        zext_ln350_2_reg_2906_pp0_iter1_reg[9 : 0] <= zext_ln350_2_reg_2906[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln336_reg_2902_pp0_iter2_reg <= icmp_ln336_reg_2902_pp0_iter1_reg;
        rq_buf_1_load_reg_3013 <= rq_buf_1_q0;
        rq_buf_2_load_reg_3023 <= rq_buf_2_q0;
        rq_buf_4_load_reg_3050 <= rq_buf_4_q0;
        rq_buf_6_load_reg_3077 <= rq_buf_6_q0;
        tmp_10_reg_3154 <= {{vq_buf_2_q0[23:16]}};
        tmp_12_reg_3164 <= {{vq_buf_4_q0[23:16]}};
        tmp_14_reg_3174 <= {{vq_buf_6_q0[23:16]}};
        tmp_17_reg_3189 <= {{vq_buf_1_q0[31:24]}};
        tmp_18_reg_3194 <= {{vq_buf_2_q0[31:24]}};
        tmp_1_reg_3109 <= {{vq_buf_1_q0[15:8]}};
        tmp_20_reg_3204 <= {{vq_buf_4_q0[31:24]}};
        tmp_22_reg_3214 <= {{vq_buf_6_q0[31:24]}};
        tmp_25_reg_3229 <= {{vq_buf_1_q0[39:32]}};
        tmp_26_reg_3234 <= {{vq_buf_2_q0[39:32]}};
        tmp_28_reg_3244 <= {{vq_buf_4_q0[39:32]}};
        tmp_2_reg_3114 <= {{vq_buf_2_q0[15:8]}};
        tmp_30_reg_3254 <= {{vq_buf_6_q0[39:32]}};
        tmp_33_reg_3269 <= {{vq_buf_1_q0[47:40]}};
        tmp_34_reg_3274 <= {{vq_buf_2_q0[47:40]}};
        tmp_36_reg_3284 <= {{vq_buf_4_q0[47:40]}};
        tmp_38_reg_3294 <= {{vq_buf_6_q0[47:40]}};
        tmp_41_reg_3309 <= {{vq_buf_1_q0[55:48]}};
        tmp_42_reg_3314 <= {{vq_buf_2_q0[55:48]}};
        tmp_44_reg_3324 <= {{vq_buf_4_q0[55:48]}};
        tmp_46_reg_3334 <= {{vq_buf_6_q0[55:48]}};
        tmp_49_reg_3349 <= {{vq_buf_1_q0[63:56]}};
        tmp_4_reg_3124 <= {{vq_buf_4_q0[15:8]}};
        tmp_50_reg_3354 <= {{vq_buf_2_q0[63:56]}};
        tmp_52_reg_3364 <= {{vq_buf_4_q0[63:56]}};
        tmp_54_reg_3374 <= {{vq_buf_6_q0[63:56]}};
        tmp_6_reg_3134 <= {{vq_buf_6_q0[15:8]}};
        tmp_9_reg_3149 <= {{vq_buf_1_q0[23:16]}};
        trunc_ln350_1_reg_3018 <= trunc_ln350_1_fu_740_p1;
        trunc_ln350_2_reg_3028 <= trunc_ln350_2_fu_744_p1;
        trunc_ln350_4_reg_3055 <= trunc_ln350_4_fu_760_p1;
        trunc_ln350_6_reg_3082 <= trunc_ln350_6_fu_776_p1;
        zext_ln336_reg_2951_pp0_iter2_reg[7 : 0] <= zext_ln336_reg_2951[7 : 0];
        zext_ln350_2_reg_2906_pp0_iter2_reg[9 : 0] <= zext_ln350_2_reg_2906_pp0_iter1_reg[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter3_fsm_state4)) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln336_reg_2902_pp0_iter3_reg <= icmp_ln336_reg_2902_pp0_iter2_reg;
        mul_res_10_reg_3409 <= mul_res_10_fu_1524_p2;
        mul_res_12_reg_3414 <= mul_res_12_fu_1533_p2;
        mul_res_14_reg_3419 <= mul_res_14_fu_1542_p2;
        mul_res_17_reg_3424 <= mul_res_17_fu_1551_p2;
        mul_res_18_reg_3429 <= mul_res_18_fu_1560_p2;
        mul_res_1_reg_3384 <= mul_res_1_fu_1470_p2;
        mul_res_20_reg_3434 <= mul_res_20_fu_1569_p2;
        mul_res_22_reg_3439 <= mul_res_22_fu_1578_p2;
        mul_res_25_reg_3444 <= mul_res_25_fu_1587_p2;
        mul_res_26_reg_3449 <= mul_res_26_fu_1596_p2;
        mul_res_28_reg_3454 <= mul_res_28_fu_1605_p2;
        mul_res_2_reg_3389 <= mul_res_2_fu_1482_p2;
        mul_res_30_reg_3459 <= mul_res_30_fu_1614_p2;
        mul_res_33_reg_3464 <= mul_res_33_fu_1623_p2;
        mul_res_34_reg_3469 <= mul_res_34_fu_1632_p2;
        mul_res_36_reg_3474 <= mul_res_36_fu_1641_p2;
        mul_res_38_reg_3479 <= mul_res_38_fu_1650_p2;
        mul_res_41_reg_3484 <= mul_res_41_fu_1659_p2;
        mul_res_42_reg_3489 <= mul_res_42_fu_1668_p2;
        mul_res_44_reg_3494 <= mul_res_44_fu_1677_p2;
        mul_res_46_reg_3499 <= mul_res_46_fu_1686_p2;
        mul_res_49_reg_3504 <= mul_res_49_fu_1695_p2;
        mul_res_4_reg_3394 <= mul_res_4_fu_1494_p2;
        mul_res_50_reg_3509 <= mul_res_50_fu_1704_p2;
        mul_res_52_reg_3514 <= mul_res_52_fu_1713_p2;
        mul_res_54_reg_3519 <= mul_res_54_fu_1722_p2;
        mul_res_57_reg_3524 <= mul_res_57_fu_1731_p2;
        mul_res_58_reg_3529 <= mul_res_58_fu_1740_p2;
        mul_res_60_reg_3534 <= mul_res_60_fu_1749_p2;
        mul_res_62_reg_3539 <= mul_res_62_fu_1758_p2;
        mul_res_6_reg_3399 <= mul_res_6_fu_1506_p2;
        mul_res_9_reg_3404 <= mul_res_9_fu_1515_p2;
        zext_ln336_reg_2951_pp0_iter3_reg[7 : 0] <= zext_ln336_reg_2951_pp0_iter2_reg[7 : 0];
        zext_ln350_2_reg_2906_pp0_iter3_reg[9 : 0] <= zext_ln350_2_reg_2906_pp0_iter2_reg[9 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter4_fsm_state5)) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        icmp_ln336_reg_2902_pp0_iter4_reg <= icmp_ln336_reg_2902_pp0_iter3_reg;
        zext_ln336_reg_2951_pp0_iter4_reg[7 : 0] <= zext_ln336_reg_2951_pp0_iter3_reg[7 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter7_fsm_state8)) begin
        icmp_ln336_reg_2902_pp0_iter7_reg <= icmp_ln336_reg_2902_pp0_iter6_reg;
        shl_ln360_1_reg_3879 <= shl_ln360_1_fu_2342_p2;
        shl_ln360_2_reg_3884 <= shl_ln360_2_fu_2351_p2;
        shl_ln360_3_reg_3889 <= shl_ln360_3_fu_2360_p2;
        shl_ln360_4_reg_3894 <= shl_ln360_4_fu_2369_p2;
        shl_ln360_5_reg_3899 <= shl_ln360_5_fu_2378_p2;
        shl_ln360_6_reg_3904 <= shl_ln360_6_fu_2387_p2;
        shl_ln360_7_reg_3909 <= shl_ln360_7_fu_2396_p2;
        shl_ln360_reg_3874 <= shl_ln360_fu_2333_p2;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state2_blk = 1'b0;

assign ap_ST_iter2_fsm_state3_blk = 1'b0;

assign ap_ST_iter3_fsm_state4_blk = 1'b0;

assign ap_ST_iter4_fsm_state5_blk = 1'b0;

assign ap_ST_iter5_fsm_state6_blk = 1'b0;

assign ap_ST_iter6_fsm_state7_blk = 1'b0;

assign ap_ST_iter7_fsm_state8_blk = 1'b0;

assign ap_ST_iter8_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln336_fu_678_p2 == 1'd1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (ap_loop_exit_ready_pp0_iter8_reg == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state0) & (1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_st_1 = 8'd0;
    end else begin
        ap_sig_allocacmp_st_1 = st_fu_324;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_1_out_ap_vld = 1'b1;
    end else begin
        psum_vec_1_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_2_out_ap_vld = 1'b1;
    end else begin
        psum_vec_2_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_3_out_ap_vld = 1'b1;
    end else begin
        psum_vec_3_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_4_out_ap_vld = 1'b1;
    end else begin
        psum_vec_4_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_5_out_ap_vld = 1'b1;
    end else begin
        psum_vec_5_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_6_out_ap_vld = 1'b1;
    end else begin
        psum_vec_6_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_7_out_ap_vld = 1'b1;
    end else begin
        psum_vec_7_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1))) begin
        psum_vec_out_ap_vld = 1'b1;
    end else begin
        psum_vec_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_1_ce0 = 1'b1;
    end else begin
        rq_buf_1_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_2_ce0 = 1'b1;
    end else begin
        rq_buf_2_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_3_ce0 = 1'b1;
    end else begin
        rq_buf_3_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_4_ce0 = 1'b1;
    end else begin
        rq_buf_4_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_5_ce0 = 1'b1;
    end else begin
        rq_buf_5_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_6_ce0 = 1'b1;
    end else begin
        rq_buf_6_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_7_ce0 = 1'b1;
    end else begin
        rq_buf_7_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        rq_buf_ce0 = 1'b1;
    end else begin
        rq_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_iter5_fsm_state6)) begin
        rs_buf_ce0 = 1'b1;
    end else begin
        rs_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_0_ce0 = 1'b1;
    end else begin
        vq_buf_0_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_1_ce0 = 1'b1;
    end else begin
        vq_buf_1_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_2_ce0 = 1'b1;
    end else begin
        vq_buf_2_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_3_ce0 = 1'b1;
    end else begin
        vq_buf_3_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_4_ce0 = 1'b1;
    end else begin
        vq_buf_4_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_5_ce0 = 1'b1;
    end else begin
        vq_buf_5_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_6_ce0 = 1'b1;
    end else begin
        vq_buf_6_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) | (1'b1 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
        vq_buf_7_ce0 = 1'b1;
    end else begin
        vq_buf_7_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter6_fsm_state7) | (1'b1 == ap_CS_iter4_fsm_state5) | (1'b1 == ap_CS_iter5_fsm_state6))) begin
        vs_buf_ce0 = 1'b1;
    end else begin
        vs_buf_ce0 = 1'b0;
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
            if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if (((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
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
            if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
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
            if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter2_fsm_state3)) begin
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
            if ((1'b1 == ap_CS_iter3_fsm_state4)) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter3_fsm_state4)) begin
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
            if ((1'b1 == ap_CS_iter4_fsm_state5)) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter4_fsm_state5)) begin
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
            if ((1'b1 == ap_CS_iter5_fsm_state6)) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter5_fsm_state6)) begin
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
            if ((1'b1 == ap_CS_iter6_fsm_state7)) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter6_fsm_state7)) begin
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
            if ((1'b0 == ap_CS_iter7_fsm_state8)) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state0;
            end else if (((1'b1 == ap_CS_iter7_fsm_state8) | ((1'b1 == ap_CS_iter8_fsm_state9) & (icmp_ln336_reg_2902_pp0_iter7_reg == 1'd1)))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end else begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end
        end
        ap_ST_iter8_fsm_state0 : begin
            if ((1'b1 == ap_CS_iter7_fsm_state8)) begin
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

assign add_ln336_fu_684_p2 = (ap_sig_allocacmp_st_1 + 8'd1);

assign add_ln350_fu_694_p2 = (zext_ln350 + zext_ln350_1_fu_690_p1);

assign add_ln352_12_fu_1902_p2 = ($signed(sext_ln352_10_fu_1899_p1) + $signed(sext_ln352_9_fu_1896_p1));

assign add_ln352_13_fu_2070_p2 = ($signed(sext_ln352_11_fu_2067_p1) + $signed(sext_ln352_8_fu_2064_p1));

assign add_ln352_16_fu_1914_p2 = ($signed(sext_ln352_13_fu_1911_p1) + $signed(sext_ln352_12_fu_1908_p1));

assign add_ln352_19_fu_1926_p2 = ($signed(sext_ln352_16_fu_1923_p1) + $signed(sext_ln352_15_fu_1920_p1));

assign add_ln352_20_fu_2082_p2 = ($signed(sext_ln352_17_fu_2079_p1) + $signed(sext_ln352_14_fu_2076_p1));

assign add_ln352_23_fu_1938_p2 = ($signed(sext_ln352_19_fu_1935_p1) + $signed(sext_ln352_18_fu_1932_p1));

assign add_ln352_26_fu_1950_p2 = ($signed(sext_ln352_22_fu_1947_p1) + $signed(sext_ln352_21_fu_1944_p1));

assign add_ln352_27_fu_2094_p2 = ($signed(sext_ln352_23_fu_2091_p1) + $signed(sext_ln352_20_fu_2088_p1));

assign add_ln352_2_fu_1866_p2 = ($signed(sext_ln352_1_fu_1863_p1) + $signed(sext_ln352_fu_1860_p1));

assign add_ln352_30_fu_1962_p2 = ($signed(sext_ln352_25_fu_1959_p1) + $signed(sext_ln352_24_fu_1956_p1));

assign add_ln352_33_fu_1974_p2 = ($signed(sext_ln352_28_fu_1971_p1) + $signed(sext_ln352_27_fu_1968_p1));

assign add_ln352_34_fu_2106_p2 = ($signed(sext_ln352_29_fu_2103_p1) + $signed(sext_ln352_26_fu_2100_p1));

assign add_ln352_37_fu_1986_p2 = ($signed(sext_ln352_31_fu_1983_p1) + $signed(sext_ln352_30_fu_1980_p1));

assign add_ln352_40_fu_1998_p2 = ($signed(sext_ln352_34_fu_1995_p1) + $signed(sext_ln352_33_fu_1992_p1));

assign add_ln352_41_fu_2118_p2 = ($signed(sext_ln352_35_fu_2115_p1) + $signed(sext_ln352_32_fu_2112_p1));

assign add_ln352_44_fu_2010_p2 = ($signed(sext_ln352_37_fu_2007_p1) + $signed(sext_ln352_36_fu_2004_p1));

assign add_ln352_47_fu_2022_p2 = ($signed(sext_ln352_40_fu_2019_p1) + $signed(sext_ln352_39_fu_2016_p1));

assign add_ln352_48_fu_2130_p2 = ($signed(sext_ln352_41_fu_2127_p1) + $signed(sext_ln352_38_fu_2124_p1));

assign add_ln352_51_fu_2034_p2 = ($signed(sext_ln352_43_fu_2031_p1) + $signed(sext_ln352_42_fu_2028_p1));

assign add_ln352_54_fu_2046_p2 = ($signed(sext_ln352_46_fu_2043_p1) + $signed(sext_ln352_45_fu_2040_p1));

assign add_ln352_55_fu_2142_p2 = ($signed(sext_ln352_47_fu_2139_p1) + $signed(sext_ln352_44_fu_2136_p1));

assign add_ln352_5_fu_1878_p2 = ($signed(sext_ln352_4_fu_1875_p1) + $signed(sext_ln352_3_fu_1872_p1));

assign add_ln352_6_fu_2058_p2 = ($signed(sext_ln352_5_fu_2055_p1) + $signed(sext_ln352_2_fu_2052_p1));

assign add_ln352_9_fu_1890_p2 = ($signed(sext_ln352_7_fu_1887_p1) + $signed(sext_ln352_6_fu_1884_p1));

assign add_ln360_10_fu_2260_p2 = (zext_ln360_11_fu_2256_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_11_fu_2451_p2 = (shl_ln360_5_reg_3899 + psum_vec_5_fu_312);

assign add_ln360_12_fu_2280_p2 = (zext_ln360_13_fu_2276_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_13_fu_2456_p2 = (shl_ln360_6_reg_3904 + psum_vec_6_fu_316);

assign add_ln360_14_fu_2300_p2 = (zext_ln360_15_fu_2296_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_15_fu_2461_p2 = (shl_ln360_7_reg_3909 + psum_vec_7_fu_320);

assign add_ln360_1_fu_2426_p2 = (shl_ln360_reg_3874 + psum_vec_fu_292);

assign add_ln360_2_fu_2180_p2 = (zext_ln360_3_fu_2176_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_3_fu_2431_p2 = (shl_ln360_1_reg_3879 + psum_vec_1_fu_296);

assign add_ln360_4_fu_2200_p2 = (zext_ln360_5_fu_2196_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_5_fu_2436_p2 = (shl_ln360_2_reg_3884 + psum_vec_2_fu_300);

assign add_ln360_6_fu_2220_p2 = (zext_ln360_7_fu_2216_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_7_fu_2441_p2 = (shl_ln360_3_reg_3889 + psum_vec_3_fu_304);

assign add_ln360_8_fu_2240_p2 = (zext_ln360_9_fu_2236_p1 + zext_ln360_fu_2152_p1);

assign add_ln360_9_fu_2446_p2 = (shl_ln360_4_reg_3894 + psum_vec_4_fu_308);

assign add_ln360_fu_2160_p2 = (zext_ln360_1_fu_2156_p1 + zext_ln360_fu_2152_p1);

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

assign ap_CS_iter8_fsm_state0 = ap_CS_iter8_fsm[32'd0];

assign ap_CS_iter8_fsm_state9 = ap_CS_iter8_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign grp_fu_2538_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2547_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2556_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2565_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2574_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2583_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2592_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2601_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2610_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2619_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2628_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2637_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2646_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2655_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2664_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2673_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2682_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2691_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2700_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2709_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2718_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2727_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2736_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2745_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2754_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2763_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2772_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2781_p1 = sext_ln350_21_fu_784_p1;

assign grp_fu_2790_p1 = sext_ln350_fu_732_p1;

assign grp_fu_2799_p1 = sext_ln350_9_fu_752_p1;

assign grp_fu_2808_p1 = sext_ln350_15_fu_768_p1;

assign grp_fu_2817_p1 = sext_ln350_21_fu_784_p1;

assign icmp_ln336_fu_678_p2 = ((ap_sig_allocacmp_st_1 == valid_st) ? 1'b1 : 1'b0);

assign mul_res_10_fu_1524_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_12_fu_1533_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_14_fu_1542_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_17_fu_1551_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_18_fu_1560_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_1_fu_1470_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_20_fu_1569_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_22_fu_1578_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_25_fu_1587_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_26_fu_1596_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_28_fu_1605_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_2_fu_1482_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_30_fu_1614_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_33_fu_1623_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_34_fu_1632_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_36_fu_1641_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_38_fu_1650_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_41_fu_1659_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_42_fu_1668_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_44_fu_1677_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_46_fu_1686_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_49_fu_1695_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_4_fu_1494_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_50_fu_1704_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_52_fu_1713_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_54_fu_1722_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_57_fu_1731_p1 = sext_ln350_3_fu_1464_p1;

assign mul_res_58_fu_1740_p1 = sext_ln350_6_fu_1476_p1;

assign mul_res_60_fu_1749_p1 = sext_ln350_12_fu_1488_p1;

assign mul_res_62_fu_1758_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_6_fu_1506_p1 = sext_ln350_18_fu_1500_p1;

assign mul_res_9_fu_1515_p1 = sext_ln350_3_fu_1464_p1;

assign psum_vec_1_out = psum_vec_1_fu_296;

assign psum_vec_2_out = psum_vec_2_fu_300;

assign psum_vec_3_out = psum_vec_3_fu_304;

assign psum_vec_4_out = psum_vec_4_fu_308;

assign psum_vec_5_out = psum_vec_5_fu_312;

assign psum_vec_6_out = psum_vec_6_fu_316;

assign psum_vec_7_out = psum_vec_7_fu_320;

assign psum_vec_out = psum_vec_fu_292;

assign rq_buf_1_address0 = zext_ln336_fu_717_p1;

assign rq_buf_2_address0 = zext_ln336_fu_717_p1;

assign rq_buf_3_address0 = zext_ln336_fu_717_p1;

assign rq_buf_4_address0 = zext_ln336_fu_717_p1;

assign rq_buf_5_address0 = zext_ln336_fu_717_p1;

assign rq_buf_6_address0 = zext_ln336_fu_717_p1;

assign rq_buf_7_address0 = zext_ln336_fu_717_p1;

assign rq_buf_address0 = zext_ln336_fu_717_p1;

assign rs_buf_address0 = zext_ln336_reg_2951_pp0_iter4_reg;

assign sext_ln350_109_fu_2321_p1 = $signed(add_ln352_41_reg_3819);

assign sext_ln350_126_fu_2324_p1 = $signed(add_ln352_48_reg_3824);

assign sext_ln350_12_fu_1488_p1 = $signed(rq_buf_4_load_reg_3050);

assign sext_ln350_15_fu_768_p1 = $signed(rq_buf_5_q0);

assign sext_ln350_18_fu_1500_p1 = $signed(rq_buf_6_load_reg_3077);

assign sext_ln350_21_fu_784_p1 = $signed(rq_buf_7_q0);

assign sext_ln350_24_fu_2306_p1 = $signed(add_ln352_6_reg_3794);

assign sext_ln350_3_fu_1464_p1 = $signed(rq_buf_1_load_reg_3013);

assign sext_ln350_41_fu_2309_p1 = $signed(add_ln352_13_reg_3799);

assign sext_ln350_58_fu_2312_p1 = $signed(add_ln352_20_reg_3804);

assign sext_ln350_6_fu_1476_p1 = $signed(rq_buf_2_load_reg_3023);

assign sext_ln350_75_fu_2315_p1 = $signed(add_ln352_27_reg_3809);

assign sext_ln350_92_fu_2318_p1 = $signed(add_ln352_34_reg_3814);

assign sext_ln350_9_fu_752_p1 = $signed(rq_buf_3_q0);

assign sext_ln350_fu_732_p1 = $signed(rq_buf_q0);

assign sext_ln352_10_fu_1899_p1 = grp_fu_2601_p3;

assign sext_ln352_11_fu_2067_p1 = $signed(add_ln352_12_reg_3724);

assign sext_ln352_12_fu_1908_p1 = grp_fu_2610_p3;

assign sext_ln352_13_fu_1911_p1 = grp_fu_2619_p3;

assign sext_ln352_14_fu_2076_p1 = $signed(add_ln352_16_reg_3729);

assign sext_ln352_15_fu_1920_p1 = grp_fu_2628_p3;

assign sext_ln352_16_fu_1923_p1 = grp_fu_2637_p3;

assign sext_ln352_17_fu_2079_p1 = $signed(add_ln352_19_reg_3734);

assign sext_ln352_18_fu_1932_p1 = grp_fu_2646_p3;

assign sext_ln352_19_fu_1935_p1 = grp_fu_2655_p3;

assign sext_ln352_1_fu_1863_p1 = grp_fu_2547_p3;

assign sext_ln352_20_fu_2088_p1 = $signed(add_ln352_23_reg_3739);

assign sext_ln352_21_fu_1944_p1 = grp_fu_2664_p3;

assign sext_ln352_22_fu_1947_p1 = grp_fu_2673_p3;

assign sext_ln352_23_fu_2091_p1 = $signed(add_ln352_26_reg_3744);

assign sext_ln352_24_fu_1956_p1 = grp_fu_2682_p3;

assign sext_ln352_25_fu_1959_p1 = grp_fu_2691_p3;

assign sext_ln352_26_fu_2100_p1 = $signed(add_ln352_30_reg_3749);

assign sext_ln352_27_fu_1968_p1 = grp_fu_2700_p3;

assign sext_ln352_28_fu_1971_p1 = grp_fu_2709_p3;

assign sext_ln352_29_fu_2103_p1 = $signed(add_ln352_33_reg_3754);

assign sext_ln352_2_fu_2052_p1 = $signed(add_ln352_2_reg_3709);

assign sext_ln352_30_fu_1980_p1 = grp_fu_2718_p3;

assign sext_ln352_31_fu_1983_p1 = grp_fu_2727_p3;

assign sext_ln352_32_fu_2112_p1 = $signed(add_ln352_37_reg_3759);

assign sext_ln352_33_fu_1992_p1 = grp_fu_2736_p3;

assign sext_ln352_34_fu_1995_p1 = grp_fu_2745_p3;

assign sext_ln352_35_fu_2115_p1 = $signed(add_ln352_40_reg_3764);

assign sext_ln352_36_fu_2004_p1 = grp_fu_2754_p3;

assign sext_ln352_37_fu_2007_p1 = grp_fu_2763_p3;

assign sext_ln352_38_fu_2124_p1 = $signed(add_ln352_44_reg_3769);

assign sext_ln352_39_fu_2016_p1 = grp_fu_2772_p3;

assign sext_ln352_3_fu_1872_p1 = grp_fu_2556_p3;

assign sext_ln352_40_fu_2019_p1 = grp_fu_2781_p3;

assign sext_ln352_41_fu_2127_p1 = $signed(add_ln352_47_reg_3774);

assign sext_ln352_42_fu_2028_p1 = grp_fu_2790_p3;

assign sext_ln352_43_fu_2031_p1 = grp_fu_2799_p3;

assign sext_ln352_44_fu_2136_p1 = $signed(add_ln352_51_reg_3779);

assign sext_ln352_45_fu_2040_p1 = grp_fu_2808_p3;

assign sext_ln352_46_fu_2043_p1 = grp_fu_2817_p3;

assign sext_ln352_47_fu_2139_p1 = $signed(add_ln352_54_reg_3784);

assign sext_ln352_4_fu_1875_p1 = grp_fu_2565_p3;

assign sext_ln352_5_fu_2055_p1 = $signed(add_ln352_5_reg_3714);

assign sext_ln352_6_fu_1884_p1 = grp_fu_2574_p3;

assign sext_ln352_7_fu_1887_p1 = grp_fu_2583_p3;

assign sext_ln352_8_fu_2064_p1 = $signed(add_ln352_9_reg_3719);

assign sext_ln352_9_fu_1896_p1 = grp_fu_2592_p3;

assign sext_ln352_fu_1860_p1 = grp_fu_2538_p3;

assign sext_ln356_fu_2327_p1 = $signed(add_ln352_55_reg_3829);

assign shl_ln360_1_fu_2342_p2 = sext_ln350_41_fu_2309_p1 << zext_ln360_4_fu_2339_p1;

assign shl_ln360_2_fu_2351_p2 = sext_ln350_58_fu_2312_p1 << zext_ln360_6_fu_2348_p1;

assign shl_ln360_3_fu_2360_p2 = sext_ln350_75_fu_2315_p1 << zext_ln360_8_fu_2357_p1;

assign shl_ln360_4_fu_2369_p2 = sext_ln350_92_fu_2318_p1 << zext_ln360_10_fu_2366_p1;

assign shl_ln360_5_fu_2378_p2 = sext_ln350_109_fu_2321_p1 << zext_ln360_12_fu_2375_p1;

assign shl_ln360_6_fu_2387_p2 = sext_ln350_126_fu_2324_p1 << zext_ln360_14_fu_2384_p1;

assign shl_ln360_7_fu_2396_p2 = sext_ln356_fu_2327_p1 << zext_ln360_16_fu_2393_p1;

assign shl_ln360_fu_2333_p2 = sext_ln350_24_fu_2306_p1 << zext_ln360_2_fu_2330_p1;

assign tmp_11_fu_922_p4 = {{vq_buf_3_q0[23:16]}};

assign tmp_13_fu_946_p4 = {{vq_buf_5_q0[23:16]}};

assign tmp_15_fu_970_p4 = {{vq_buf_7_q0[23:16]}};

assign tmp_16_fu_984_p4 = {{vq_buf_0_q0[31:24]}};

assign tmp_19_fu_1018_p4 = {{vq_buf_3_q0[31:24]}};

assign tmp_21_fu_1042_p4 = {{vq_buf_5_q0[31:24]}};

assign tmp_23_fu_1066_p4 = {{vq_buf_7_q0[31:24]}};

assign tmp_24_fu_1080_p4 = {{vq_buf_0_q0[39:32]}};

assign tmp_27_fu_1114_p4 = {{vq_buf_3_q0[39:32]}};

assign tmp_29_fu_1138_p4 = {{vq_buf_5_q0[39:32]}};

assign tmp_31_fu_1162_p4 = {{vq_buf_7_q0[39:32]}};

assign tmp_32_fu_1176_p4 = {{vq_buf_0_q0[47:40]}};

assign tmp_35_fu_1210_p4 = {{vq_buf_3_q0[47:40]}};

assign tmp_37_fu_1234_p4 = {{vq_buf_5_q0[47:40]}};

assign tmp_39_fu_1258_p4 = {{vq_buf_7_q0[47:40]}};

assign tmp_3_fu_826_p4 = {{vq_buf_3_q0[15:8]}};

assign tmp_40_fu_1272_p4 = {{vq_buf_0_q0[55:48]}};

assign tmp_43_fu_1306_p4 = {{vq_buf_3_q0[55:48]}};

assign tmp_45_fu_1330_p4 = {{vq_buf_5_q0[55:48]}};

assign tmp_47_fu_1354_p4 = {{vq_buf_7_q0[55:48]}};

assign tmp_48_fu_1368_p4 = {{vq_buf_0_q0[63:56]}};

assign tmp_51_fu_1402_p4 = {{vq_buf_3_q0[63:56]}};

assign tmp_53_fu_1426_p4 = {{vq_buf_5_q0[63:56]}};

assign tmp_55_fu_1450_p4 = {{vq_buf_7_q0[63:56]}};

assign tmp_5_fu_850_p4 = {{vq_buf_5_q0[15:8]}};

assign tmp_7_fu_874_p4 = {{vq_buf_7_q0[15:8]}};

assign tmp_8_fu_888_p4 = {{vq_buf_0_q0[23:16]}};

assign tmp_s_fu_792_p4 = {{vq_buf_0_q0[15:8]}};

assign trunc_ln350_1_fu_740_p1 = vq_buf_1_q0[7:0];

assign trunc_ln350_2_fu_744_p1 = vq_buf_2_q0[7:0];

assign trunc_ln350_3_fu_748_p1 = vq_buf_3_q0[7:0];

assign trunc_ln350_4_fu_760_p1 = vq_buf_4_q0[7:0];

assign trunc_ln350_5_fu_764_p1 = vq_buf_5_q0[7:0];

assign trunc_ln350_6_fu_776_p1 = vq_buf_6_q0[7:0];

assign trunc_ln350_7_fu_780_p1 = vq_buf_7_q0[7:0];

assign trunc_ln350_fu_728_p1 = vq_buf_0_q0[7:0];

assign v_shift_1_fu_2166_p4 = {{vs_buf_q0[7:4]}};

assign v_shift_2_fu_2186_p4 = {{vs_buf_q0[11:8]}};

assign v_shift_3_fu_2206_p4 = {{vs_buf_q0[15:12]}};

assign v_shift_4_fu_2226_p4 = {{vs_buf_q0[19:16]}};

assign v_shift_5_fu_2246_p4 = {{vs_buf_q0[23:20]}};

assign v_shift_6_fu_2266_p4 = {{vs_buf_q0[27:24]}};

assign v_shift_7_fu_2286_p4 = {{vs_buf_q0[31:28]}};

assign v_shift_fu_2148_p1 = vs_buf_q0[3:0];

assign vq_buf_0_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_1_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_2_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_3_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_4_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_5_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_6_address0 = zext_ln350_2_fu_700_p1;

assign vq_buf_7_address0 = zext_ln350_2_fu_700_p1;

assign vs_buf_address0 = zext_ln350_2_reg_2906_pp0_iter3_reg;

assign zext_ln336_fu_717_p1 = st_1_reg_2897;

assign zext_ln350_1_fu_690_p1 = ap_sig_allocacmp_st_1;

assign zext_ln350_2_fu_700_p1 = add_ln350_fu_694_p2;

assign zext_ln360_10_fu_2366_p1 = add_ln360_8_reg_3854;

assign zext_ln360_11_fu_2256_p1 = v_shift_5_fu_2246_p4;

assign zext_ln360_12_fu_2375_p1 = add_ln360_10_reg_3859;

assign zext_ln360_13_fu_2276_p1 = v_shift_6_fu_2266_p4;

assign zext_ln360_14_fu_2384_p1 = add_ln360_12_reg_3864;

assign zext_ln360_15_fu_2296_p1 = v_shift_7_fu_2286_p4;

assign zext_ln360_16_fu_2393_p1 = add_ln360_14_reg_3869;

assign zext_ln360_1_fu_2156_p1 = v_shift_fu_2148_p1;

assign zext_ln360_2_fu_2330_p1 = add_ln360_reg_3834;

assign zext_ln360_3_fu_2176_p1 = v_shift_1_fu_2166_p4;

assign zext_ln360_4_fu_2339_p1 = add_ln360_2_reg_3839;

assign zext_ln360_5_fu_2196_p1 = v_shift_2_fu_2186_p4;

assign zext_ln360_6_fu_2348_p1 = add_ln360_4_reg_3844;

assign zext_ln360_7_fu_2216_p1 = v_shift_3_fu_2206_p4;

assign zext_ln360_8_fu_2357_p1 = add_ln360_6_reg_3849;

assign zext_ln360_9_fu_2236_p1 = v_shift_4_fu_2226_p4;

assign zext_ln360_fu_2152_p1 = rs_buf_q0;

always @ (posedge ap_clk) begin
    zext_ln350_2_reg_2906[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln350_2_reg_2906_pp0_iter1_reg[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln350_2_reg_2906_pp0_iter2_reg[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln350_2_reg_2906_pp0_iter3_reg[63:10] <= 54'b000000000000000000000000000000000000000000000000000000;
    zext_ln336_reg_2951[63:8] <= 56'b00000000000000000000000000000000000000000000000000000000;
    zext_ln336_reg_2951_pp0_iter2_reg[63:8] <= 56'b00000000000000000000000000000000000000000000000000000000;
    zext_ln336_reg_2951_pp0_iter3_reg[63:8] <= 56'b00000000000000000000000000000000000000000000000000000000;
    zext_ln336_reg_2951_pp0_iter4_reg[63:8] <= 56'b00000000000000000000000000000000000000000000000000000000;
end

endmodule //RV_GEMM_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RV_GEMM_shared_rv_bmm_head (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        num_tokens,
        valid_st,
        vq_buf_0_address0,
        vq_buf_0_ce0,
        vq_buf_0_q0,
        vq_buf_1_address0,
        vq_buf_1_ce0,
        vq_buf_1_q0,
        vq_buf_2_address0,
        vq_buf_2_ce0,
        vq_buf_2_q0,
        vq_buf_3_address0,
        vq_buf_3_ce0,
        vq_buf_3_q0,
        vq_buf_4_address0,
        vq_buf_4_ce0,
        vq_buf_4_q0,
        vq_buf_5_address0,
        vq_buf_5_ce0,
        vq_buf_5_q0,
        vq_buf_6_address0,
        vq_buf_6_ce0,
        vq_buf_6_q0,
        vq_buf_7_address0,
        vq_buf_7_ce0,
        vq_buf_7_q0,
        vs_buf_address0,
        vs_buf_ce0,
        vs_buf_q0,
        rq_stream_TDATA,
        rq_stream_TVALID,
        rq_stream_TREADY,
        rs_stream_TDATA,
        rs_stream_TVALID,
        rs_stream_TREADY,
        aq_stream_TDATA,
        aq_stream_TVALID,
        aq_stream_TREADY,
        as_stream_TDATA,
        as_stream_TVALID,
        as_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 9'd1;
parameter    ap_ST_fsm_state2 = 9'd2;
parameter    ap_ST_fsm_state3 = 9'd4;
parameter    ap_ST_fsm_state4 = 9'd8;
parameter    ap_ST_fsm_state5 = 9'd16;
parameter    ap_ST_fsm_state6 = 9'd32;
parameter    ap_ST_fsm_state7 = 9'd64;
parameter    ap_ST_fsm_state8 = 9'd128;
parameter    ap_ST_fsm_state9 = 9'd256;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [10:0] num_tokens;
input  [7:0] valid_st;
output  [9:0] vq_buf_0_address0;
output   vq_buf_0_ce0;
input  [63:0] vq_buf_0_q0;
output  [9:0] vq_buf_1_address0;
output   vq_buf_1_ce0;
input  [63:0] vq_buf_1_q0;
output  [9:0] vq_buf_2_address0;
output   vq_buf_2_ce0;
input  [63:0] vq_buf_2_q0;
output  [9:0] vq_buf_3_address0;
output   vq_buf_3_ce0;
input  [63:0] vq_buf_3_q0;
output  [9:0] vq_buf_4_address0;
output   vq_buf_4_ce0;
input  [63:0] vq_buf_4_q0;
output  [9:0] vq_buf_5_address0;
output   vq_buf_5_ce0;
input  [63:0] vq_buf_5_q0;
output  [9:0] vq_buf_6_address0;
output   vq_buf_6_ce0;
input  [63:0] vq_buf_6_q0;
output  [9:0] vq_buf_7_address0;
output   vq_buf_7_ce0;
input  [63:0] vq_buf_7_q0;
output  [9:0] vs_buf_address0;
output   vs_buf_ce0;
input  [31:0] vs_buf_q0;
input  [63:0] rq_stream_TDATA;
input   rq_stream_TVALID;
output   rq_stream_TREADY;
input  [7:0] rs_stream_TDATA;
input   rs_stream_TVALID;
output   rs_stream_TREADY;
output  [63:0] aq_stream_TDATA;
output   aq_stream_TVALID;
input   aq_stream_TREADY;
output  [7:0] as_stream_TDATA;
output   as_stream_TVALID;
input   as_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg rq_stream_TREADY;
reg rs_stream_TREADY;

(* fsm_encoding = "none" *) reg   [8:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [10:0] t_2_fu_290_p2;
reg   [10:0] t_2_reg_510;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln322_fu_296_p2;
reg   [0:0] icmp_ln322_reg_515;
wire   [2:0] trunc_ln328_fu_301_p1;
reg   [2:0] trunc_ln328_reg_519;
wire    ap_CS_fsm_state5;
wire   [3:0] add_ln328_fu_311_p2;
reg   [3:0] add_ln328_reg_527;
wire   [9:0] tmp_s_fu_321_p3;
reg   [9:0] tmp_s_reg_532;
wire    ap_CS_fsm_state6;
reg   [22:0] trunc_ln_reg_537;
wire    ap_CS_fsm_state8;
reg   [22:0] trunc_ln367_1_reg_542;
reg   [22:0] trunc_ln367_2_reg_547;
reg   [22:0] trunc_ln367_3_reg_552;
reg   [22:0] trunc_ln367_4_reg_557;
reg   [22:0] trunc_ln367_5_reg_562;
reg   [22:0] trunc_ln367_6_reg_567;
reg   [22:0] trunc_ln367_7_reg_572;
reg    rq_buf_ce0;
wire   [7:0] rq_buf_q0;
reg    rq_buf_ce1;
reg    rq_buf_we1;
reg    rq_buf_1_ce0;
wire   [7:0] rq_buf_1_q0;
reg    rq_buf_1_ce1;
reg    rq_buf_1_we1;
reg    rq_buf_2_ce0;
wire   [7:0] rq_buf_2_q0;
reg    rq_buf_2_ce1;
reg    rq_buf_2_we1;
reg    rq_buf_3_ce0;
wire   [7:0] rq_buf_3_q0;
reg    rq_buf_3_ce1;
reg    rq_buf_3_we1;
reg    rq_buf_4_ce0;
wire   [7:0] rq_buf_4_q0;
reg    rq_buf_4_ce1;
reg    rq_buf_4_we1;
reg    rq_buf_5_ce0;
wire   [7:0] rq_buf_5_q0;
reg    rq_buf_5_ce1;
reg    rq_buf_5_we1;
reg    rq_buf_6_ce0;
wire   [7:0] rq_buf_6_q0;
reg    rq_buf_6_ce1;
reg    rq_buf_6_we1;
reg    rq_buf_7_ce0;
wire   [7:0] rq_buf_7_q0;
reg    rq_buf_7_ce1;
reg    rq_buf_7_we1;
reg    rs_buf_ce0;
wire   [3:0] rs_buf_q0;
reg    rs_buf_ce1;
reg    rs_buf_we1;
wire    grp_load_r_row_fu_201_ap_start;
wire    grp_load_r_row_fu_201_ap_done;
wire    grp_load_r_row_fu_201_ap_idle;
wire    grp_load_r_row_fu_201_ap_ready;
wire    grp_load_r_row_fu_201_rq_stream_TREADY;
wire    grp_load_r_row_fu_201_rs_stream_TREADY;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_0_address1;
wire    grp_load_r_row_fu_201_rq_buf_0_ce1;
wire    grp_load_r_row_fu_201_rq_buf_0_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_0_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_1_address1;
wire    grp_load_r_row_fu_201_rq_buf_1_ce1;
wire    grp_load_r_row_fu_201_rq_buf_1_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_1_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_2_address1;
wire    grp_load_r_row_fu_201_rq_buf_2_ce1;
wire    grp_load_r_row_fu_201_rq_buf_2_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_2_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_3_address1;
wire    grp_load_r_row_fu_201_rq_buf_3_ce1;
wire    grp_load_r_row_fu_201_rq_buf_3_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_3_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_4_address1;
wire    grp_load_r_row_fu_201_rq_buf_4_ce1;
wire    grp_load_r_row_fu_201_rq_buf_4_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_4_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_5_address1;
wire    grp_load_r_row_fu_201_rq_buf_5_ce1;
wire    grp_load_r_row_fu_201_rq_buf_5_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_5_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_6_address1;
wire    grp_load_r_row_fu_201_rq_buf_6_ce1;
wire    grp_load_r_row_fu_201_rq_buf_6_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_6_d1;
wire   [6:0] grp_load_r_row_fu_201_rq_buf_7_address1;
wire    grp_load_r_row_fu_201_rq_buf_7_ce1;
wire    grp_load_r_row_fu_201_rq_buf_7_we1;
wire   [7:0] grp_load_r_row_fu_201_rq_buf_7_d1;
wire   [6:0] grp_load_r_row_fu_201_rs_buf_address1;
wire    grp_load_r_row_fu_201_rs_buf_ce1;
wire    grp_load_r_row_fu_201_rs_buf_we1;
wire   [3:0] grp_load_r_row_fu_201_rs_buf_d1;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_done;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_idle;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_ready;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_0_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_0_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_1_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_1_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_2_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_2_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_3_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_3_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_4_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_4_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_5_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_5_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_6_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_6_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_7_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_7_ce0;
wire   [9:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vs_buf_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vs_buf_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_1_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_1_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_2_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_2_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_3_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_3_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_4_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_4_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_5_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_5_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_6_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_6_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_7_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_7_ce0;
wire   [6:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rs_buf_address0;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rs_buf_ce0;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_7_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_7_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_6_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_6_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_5_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_5_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_4_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_4_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_3_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_3_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_2_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_2_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_1_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_1_out_ap_vld;
wire   [39:0] grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_out;
wire    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_out_ap_vld;
wire    grp_quantize_a_vec_fu_260_ap_start;
wire    grp_quantize_a_vec_fu_260_ap_done;
wire    grp_quantize_a_vec_fu_260_ap_idle;
wire    grp_quantize_a_vec_fu_260_ap_ready;
wire   [63:0] grp_quantize_a_vec_fu_260_aq_stream_TDATA;
wire    grp_quantize_a_vec_fu_260_aq_stream_TVALID;
wire    grp_quantize_a_vec_fu_260_aq_stream_TREADY;
wire   [7:0] grp_quantize_a_vec_fu_260_as_stream_TDATA;
wire    grp_quantize_a_vec_fu_260_as_stream_TVALID;
wire    grp_quantize_a_vec_fu_260_as_stream_TREADY;
reg   [3:0] hct_reg_190;
wire    ap_CS_fsm_state9;
wire    ap_CS_fsm_state4;
reg    grp_load_r_row_fu_201_ap_start_reg;
wire    ap_CS_fsm_state3;
reg    grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start_reg;
wire    ap_CS_fsm_state7;
reg    grp_quantize_a_vec_fu_260_ap_start_reg;
reg   [10:0] t_fu_106;
wire   [0:0] icmp_ln328_fu_305_p2;
wire   [0:0] icmp_ln320_fu_284_p2;
reg   [8:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
reg    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
reg    ap_ST_fsm_state9_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 9'd1;
//#0 grp_load_r_row_fu_201_ap_start_reg = 1'b0;
//#0 grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start_reg = 1'b0;
//#0 grp_quantize_a_vec_fu_260_ap_start_reg = 1'b0;
//#0 t_fu_106 = 11'd0;
end

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_address0),
    .ce0(rq_buf_ce0),
    .q0(rq_buf_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_0_address1),
    .ce1(rq_buf_ce1),
    .we1(rq_buf_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_0_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_1_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_1_address0),
    .ce0(rq_buf_1_ce0),
    .q0(rq_buf_1_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_1_address1),
    .ce1(rq_buf_1_ce1),
    .we1(rq_buf_1_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_1_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_2_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_2_address0),
    .ce0(rq_buf_2_ce0),
    .q0(rq_buf_2_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_2_address1),
    .ce1(rq_buf_2_ce1),
    .we1(rq_buf_2_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_2_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_3_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_3_address0),
    .ce0(rq_buf_3_ce0),
    .q0(rq_buf_3_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_3_address1),
    .ce1(rq_buf_3_ce1),
    .we1(rq_buf_3_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_3_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_4_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_4_address0),
    .ce0(rq_buf_4_ce0),
    .q0(rq_buf_4_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_4_address1),
    .ce1(rq_buf_4_ce1),
    .we1(rq_buf_4_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_4_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_5_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_5_address0),
    .ce0(rq_buf_5_ce0),
    .q0(rq_buf_5_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_5_address1),
    .ce1(rq_buf_5_ce1),
    .we1(rq_buf_5_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_5_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_6_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_6_address0),
    .ce0(rq_buf_6_ce0),
    .q0(rq_buf_6_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_6_address1),
    .ce1(rq_buf_6_ce1),
    .we1(rq_buf_6_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_6_d1)
);

RV_GEMM_shared_rv_bmm_head_rq_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 8 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rq_buf_7_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_7_address0),
    .ce0(rq_buf_7_ce0),
    .q0(rq_buf_7_q0),
    .address1(grp_load_r_row_fu_201_rq_buf_7_address1),
    .ce1(rq_buf_7_ce1),
    .we1(rq_buf_7_we1),
    .d1(grp_load_r_row_fu_201_rq_buf_7_d1)
);

RV_GEMM_shared_rv_bmm_head_rs_buf_RAM_2P_LUTRAM_1R1W #(
    .DataWidth( 4 ),
    .AddressRange( 128 ),
    .AddressWidth( 7 ))
rs_buf_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rs_buf_address0),
    .ce0(rs_buf_ce0),
    .q0(rs_buf_q0),
    .address1(grp_load_r_row_fu_201_rs_buf_address1),
    .ce1(rs_buf_ce1),
    .we1(rs_buf_we1),
    .d1(grp_load_r_row_fu_201_rs_buf_d1)
);

RV_GEMM_load_r_row grp_load_r_row_fu_201(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_load_r_row_fu_201_ap_start),
    .ap_done(grp_load_r_row_fu_201_ap_done),
    .ap_idle(grp_load_r_row_fu_201_ap_idle),
    .ap_ready(grp_load_r_row_fu_201_ap_ready),
    .rq_stream_TVALID(rq_stream_TVALID),
    .rs_stream_TVALID(rs_stream_TVALID),
    .rq_stream_TDATA(rq_stream_TDATA),
    .rq_stream_TREADY(grp_load_r_row_fu_201_rq_stream_TREADY),
    .rs_stream_TDATA(rs_stream_TDATA),
    .rs_stream_TREADY(grp_load_r_row_fu_201_rs_stream_TREADY),
    .rq_buf_0_address1(grp_load_r_row_fu_201_rq_buf_0_address1),
    .rq_buf_0_ce1(grp_load_r_row_fu_201_rq_buf_0_ce1),
    .rq_buf_0_we1(grp_load_r_row_fu_201_rq_buf_0_we1),
    .rq_buf_0_d1(grp_load_r_row_fu_201_rq_buf_0_d1),
    .rq_buf_1_address1(grp_load_r_row_fu_201_rq_buf_1_address1),
    .rq_buf_1_ce1(grp_load_r_row_fu_201_rq_buf_1_ce1),
    .rq_buf_1_we1(grp_load_r_row_fu_201_rq_buf_1_we1),
    .rq_buf_1_d1(grp_load_r_row_fu_201_rq_buf_1_d1),
    .rq_buf_2_address1(grp_load_r_row_fu_201_rq_buf_2_address1),
    .rq_buf_2_ce1(grp_load_r_row_fu_201_rq_buf_2_ce1),
    .rq_buf_2_we1(grp_load_r_row_fu_201_rq_buf_2_we1),
    .rq_buf_2_d1(grp_load_r_row_fu_201_rq_buf_2_d1),
    .rq_buf_3_address1(grp_load_r_row_fu_201_rq_buf_3_address1),
    .rq_buf_3_ce1(grp_load_r_row_fu_201_rq_buf_3_ce1),
    .rq_buf_3_we1(grp_load_r_row_fu_201_rq_buf_3_we1),
    .rq_buf_3_d1(grp_load_r_row_fu_201_rq_buf_3_d1),
    .rq_buf_4_address1(grp_load_r_row_fu_201_rq_buf_4_address1),
    .rq_buf_4_ce1(grp_load_r_row_fu_201_rq_buf_4_ce1),
    .rq_buf_4_we1(grp_load_r_row_fu_201_rq_buf_4_we1),
    .rq_buf_4_d1(grp_load_r_row_fu_201_rq_buf_4_d1),
    .rq_buf_5_address1(grp_load_r_row_fu_201_rq_buf_5_address1),
    .rq_buf_5_ce1(grp_load_r_row_fu_201_rq_buf_5_ce1),
    .rq_buf_5_we1(grp_load_r_row_fu_201_rq_buf_5_we1),
    .rq_buf_5_d1(grp_load_r_row_fu_201_rq_buf_5_d1),
    .rq_buf_6_address1(grp_load_r_row_fu_201_rq_buf_6_address1),
    .rq_buf_6_ce1(grp_load_r_row_fu_201_rq_buf_6_ce1),
    .rq_buf_6_we1(grp_load_r_row_fu_201_rq_buf_6_we1),
    .rq_buf_6_d1(grp_load_r_row_fu_201_rq_buf_6_d1),
    .rq_buf_7_address1(grp_load_r_row_fu_201_rq_buf_7_address1),
    .rq_buf_7_ce1(grp_load_r_row_fu_201_rq_buf_7_ce1),
    .rq_buf_7_we1(grp_load_r_row_fu_201_rq_buf_7_we1),
    .rq_buf_7_d1(grp_load_r_row_fu_201_rq_buf_7_d1),
    .rs_buf_address1(grp_load_r_row_fu_201_rs_buf_address1),
    .rs_buf_ce1(grp_load_r_row_fu_201_rs_buf_ce1),
    .rs_buf_we1(grp_load_r_row_fu_201_rs_buf_we1),
    .rs_buf_d1(grp_load_r_row_fu_201_rs_buf_d1),
    .valid_st(valid_st)
);

RV_GEMM_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4 grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start),
    .ap_done(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_done),
    .ap_idle(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_idle),
    .ap_ready(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_ready),
    .valid_st(valid_st),
    .zext_ln350(tmp_s_reg_532),
    .vq_buf_0_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_0_address0),
    .vq_buf_0_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_0_ce0),
    .vq_buf_0_q0(vq_buf_0_q0),
    .vq_buf_1_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_1_address0),
    .vq_buf_1_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_1_ce0),
    .vq_buf_1_q0(vq_buf_1_q0),
    .vq_buf_2_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_2_address0),
    .vq_buf_2_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_2_ce0),
    .vq_buf_2_q0(vq_buf_2_q0),
    .vq_buf_3_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_3_address0),
    .vq_buf_3_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_3_ce0),
    .vq_buf_3_q0(vq_buf_3_q0),
    .vq_buf_4_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_4_address0),
    .vq_buf_4_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_4_ce0),
    .vq_buf_4_q0(vq_buf_4_q0),
    .vq_buf_5_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_5_address0),
    .vq_buf_5_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_5_ce0),
    .vq_buf_5_q0(vq_buf_5_q0),
    .vq_buf_6_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_6_address0),
    .vq_buf_6_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_6_ce0),
    .vq_buf_6_q0(vq_buf_6_q0),
    .vq_buf_7_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_7_address0),
    .vq_buf_7_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_7_ce0),
    .vq_buf_7_q0(vq_buf_7_q0),
    .vs_buf_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vs_buf_address0),
    .vs_buf_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vs_buf_ce0),
    .vs_buf_q0(vs_buf_q0),
    .rq_buf_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_address0),
    .rq_buf_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_ce0),
    .rq_buf_q0(rq_buf_q0),
    .rq_buf_1_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_1_address0),
    .rq_buf_1_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_1_ce0),
    .rq_buf_1_q0(rq_buf_1_q0),
    .rq_buf_2_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_2_address0),
    .rq_buf_2_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_2_ce0),
    .rq_buf_2_q0(rq_buf_2_q0),
    .rq_buf_3_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_3_address0),
    .rq_buf_3_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_3_ce0),
    .rq_buf_3_q0(rq_buf_3_q0),
    .rq_buf_4_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_4_address0),
    .rq_buf_4_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_4_ce0),
    .rq_buf_4_q0(rq_buf_4_q0),
    .rq_buf_5_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_5_address0),
    .rq_buf_5_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_5_ce0),
    .rq_buf_5_q0(rq_buf_5_q0),
    .rq_buf_6_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_6_address0),
    .rq_buf_6_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_6_ce0),
    .rq_buf_6_q0(rq_buf_6_q0),
    .rq_buf_7_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_7_address0),
    .rq_buf_7_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_7_ce0),
    .rq_buf_7_q0(rq_buf_7_q0),
    .rs_buf_address0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rs_buf_address0),
    .rs_buf_ce0(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rs_buf_ce0),
    .rs_buf_q0(rs_buf_q0),
    .psum_vec_7_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_7_out),
    .psum_vec_7_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_7_out_ap_vld),
    .psum_vec_6_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_6_out),
    .psum_vec_6_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_6_out_ap_vld),
    .psum_vec_5_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_5_out),
    .psum_vec_5_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_5_out_ap_vld),
    .psum_vec_4_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_4_out),
    .psum_vec_4_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_4_out_ap_vld),
    .psum_vec_3_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_3_out),
    .psum_vec_3_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_3_out_ap_vld),
    .psum_vec_2_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_2_out),
    .psum_vec_2_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_2_out_ap_vld),
    .psum_vec_1_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_1_out),
    .psum_vec_1_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_1_out_ap_vld),
    .psum_vec_out(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_out),
    .psum_vec_out_ap_vld(grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_out_ap_vld)
);

RV_GEMM_quantize_a_vec grp_quantize_a_vec_fu_260(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_quantize_a_vec_fu_260_ap_start),
    .ap_done(grp_quantize_a_vec_fu_260_ap_done),
    .ap_idle(grp_quantize_a_vec_fu_260_ap_idle),
    .ap_ready(grp_quantize_a_vec_fu_260_ap_ready),
    .p_read(trunc_ln_reg_537),
    .p_read1(trunc_ln367_1_reg_542),
    .p_read2(trunc_ln367_2_reg_547),
    .p_read3(trunc_ln367_3_reg_552),
    .p_read4(trunc_ln367_4_reg_557),
    .p_read5(trunc_ln367_5_reg_562),
    .p_read6(trunc_ln367_6_reg_567),
    .p_read7(trunc_ln367_7_reg_572),
    .aq_stream_TDATA(grp_quantize_a_vec_fu_260_aq_stream_TDATA),
    .aq_stream_TVALID(grp_quantize_a_vec_fu_260_aq_stream_TVALID),
    .aq_stream_TREADY(grp_quantize_a_vec_fu_260_aq_stream_TREADY),
    .as_stream_TDATA(grp_quantize_a_vec_fu_260_as_stream_TDATA),
    .as_stream_TVALID(grp_quantize_a_vec_fu_260_as_stream_TVALID),
    .as_stream_TREADY(grp_quantize_a_vec_fu_260_as_stream_TREADY)
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
        grp_load_r_row_fu_201_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state3)) begin
            grp_load_r_row_fu_201_ap_start_reg <= 1'b1;
        end else if ((grp_load_r_row_fu_201_ap_ready == 1'b1)) begin
            grp_load_r_row_fu_201_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_quantize_a_vec_fu_260_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state8)) begin
            grp_quantize_a_vec_fu_260_ap_start_reg <= 1'b1;
        end else if ((grp_quantize_a_vec_fu_260_ap_ready == 1'b1)) begin
            grp_quantize_a_vec_fu_260_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state6)) begin
            grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start_reg <= 1'b1;
        end else if ((grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_ready == 1'b1)) begin
            grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((grp_load_r_row_fu_201_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
        hct_reg_190 <= 4'd0;
    end else if (((grp_quantize_a_vec_fu_260_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state9))) begin
        hct_reg_190 <= add_ln328_reg_527;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        t_fu_106 <= 11'd0;
    end else if (((1'b1 == ap_CS_fsm_state5) & ((icmp_ln328_fu_305_p2 == 1'd1) | (icmp_ln322_reg_515 == 1'd0)))) begin
        t_fu_106 <= t_2_reg_510;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        add_ln328_reg_527 <= add_ln328_fu_311_p2;
        trunc_ln328_reg_519 <= trunc_ln328_fu_301_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        icmp_ln322_reg_515 <= icmp_ln322_fu_296_p2;
        t_2_reg_510 <= t_2_fu_290_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        tmp_s_reg_532[9 : 7] <= tmp_s_fu_321_p3[9 : 7];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        trunc_ln367_1_reg_542 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_1_out[39:17]}};
        trunc_ln367_2_reg_547 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_2_out[39:17]}};
        trunc_ln367_3_reg_552 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_3_out[39:17]}};
        trunc_ln367_4_reg_557 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_4_out[39:17]}};
        trunc_ln367_5_reg_562 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_5_out[39:17]}};
        trunc_ln367_6_reg_567 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_6_out[39:17]}};
        trunc_ln367_7_reg_572 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_7_out[39:17]}};
        trunc_ln_reg_537 <= {{grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_psum_vec_out[39:17]}};
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((grp_load_r_row_fu_201_ap_done == 1'b0)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

always @ (*) begin
    if ((grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_done == 1'b0)) begin
        ap_ST_fsm_state7_blk = 1'b1;
    end else begin
        ap_ST_fsm_state7_blk = 1'b0;
    end
end

assign ap_ST_fsm_state8_blk = 1'b0;

always @ (*) begin
    if ((grp_quantize_a_vec_fu_260_ap_done == 1'b0)) begin
        ap_ST_fsm_state9_blk = 1'b1;
    end else begin
        ap_ST_fsm_state9_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((icmp_ln320_fu_284_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2)) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
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
    if (((icmp_ln320_fu_284_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_1_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_1_ce0;
    end else begin
        rq_buf_1_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_1_ce1 = grp_load_r_row_fu_201_rq_buf_1_ce1;
    end else begin
        rq_buf_1_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_1_we1 = grp_load_r_row_fu_201_rq_buf_1_we1;
    end else begin
        rq_buf_1_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_2_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_2_ce0;
    end else begin
        rq_buf_2_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_2_ce1 = grp_load_r_row_fu_201_rq_buf_2_ce1;
    end else begin
        rq_buf_2_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_2_we1 = grp_load_r_row_fu_201_rq_buf_2_we1;
    end else begin
        rq_buf_2_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_3_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_3_ce0;
    end else begin
        rq_buf_3_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_3_ce1 = grp_load_r_row_fu_201_rq_buf_3_ce1;
    end else begin
        rq_buf_3_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_3_we1 = grp_load_r_row_fu_201_rq_buf_3_we1;
    end else begin
        rq_buf_3_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_4_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_4_ce0;
    end else begin
        rq_buf_4_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_4_ce1 = grp_load_r_row_fu_201_rq_buf_4_ce1;
    end else begin
        rq_buf_4_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_4_we1 = grp_load_r_row_fu_201_rq_buf_4_we1;
    end else begin
        rq_buf_4_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_5_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_5_ce0;
    end else begin
        rq_buf_5_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_5_ce1 = grp_load_r_row_fu_201_rq_buf_5_ce1;
    end else begin
        rq_buf_5_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_5_we1 = grp_load_r_row_fu_201_rq_buf_5_we1;
    end else begin
        rq_buf_5_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_6_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_6_ce0;
    end else begin
        rq_buf_6_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_6_ce1 = grp_load_r_row_fu_201_rq_buf_6_ce1;
    end else begin
        rq_buf_6_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_6_we1 = grp_load_r_row_fu_201_rq_buf_6_we1;
    end else begin
        rq_buf_6_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_7_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_7_ce0;
    end else begin
        rq_buf_7_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_7_ce1 = grp_load_r_row_fu_201_rq_buf_7_ce1;
    end else begin
        rq_buf_7_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_7_we1 = grp_load_r_row_fu_201_rq_buf_7_we1;
    end else begin
        rq_buf_7_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rq_buf_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rq_buf_ce0;
    end else begin
        rq_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_ce1 = grp_load_r_row_fu_201_rq_buf_0_ce1;
    end else begin
        rq_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_buf_we1 = grp_load_r_row_fu_201_rq_buf_0_we1;
    end else begin
        rq_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rq_stream_TREADY = grp_load_r_row_fu_201_rq_stream_TREADY;
    end else begin
        rq_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        rs_buf_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_rs_buf_ce0;
    end else begin
        rs_buf_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rs_buf_ce1 = grp_load_r_row_fu_201_rs_buf_ce1;
    end else begin
        rs_buf_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rs_buf_we1 = grp_load_r_row_fu_201_rs_buf_we1;
    end else begin
        rs_buf_we1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        rs_stream_TREADY = grp_load_r_row_fu_201_rs_stream_TREADY;
    end else begin
        rs_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((icmp_ln320_fu_284_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else if (((icmp_ln320_fu_284_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln322_fu_296_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            if (((grp_load_r_row_fu_201_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            if (((1'b1 == ap_CS_fsm_state5) & ((icmp_ln328_fu_305_p2 == 1'd1) | (icmp_ln322_reg_515 == 1'd0)))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            if (((grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state7))) begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            if (((grp_quantize_a_vec_fu_260_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state9))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln328_fu_311_p2 = (hct_reg_190 + 4'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign aq_stream_TDATA = grp_quantize_a_vec_fu_260_aq_stream_TDATA;

assign aq_stream_TVALID = grp_quantize_a_vec_fu_260_aq_stream_TVALID;

assign as_stream_TDATA = grp_quantize_a_vec_fu_260_as_stream_TDATA;

assign as_stream_TVALID = grp_quantize_a_vec_fu_260_as_stream_TVALID;

assign grp_load_r_row_fu_201_ap_start = grp_load_r_row_fu_201_ap_start_reg;

assign grp_quantize_a_vec_fu_260_ap_start = grp_quantize_a_vec_fu_260_ap_start_reg;

assign grp_quantize_a_vec_fu_260_aq_stream_TREADY = (aq_stream_TREADY & ap_CS_fsm_state9);

assign grp_quantize_a_vec_fu_260_as_stream_TREADY = (as_stream_TREADY & ap_CS_fsm_state9);

assign grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_ap_start_reg;

assign icmp_ln320_fu_284_p2 = ((t_fu_106 == 11'd1024) ? 1'b1 : 1'b0);

assign icmp_ln322_fu_296_p2 = ((t_fu_106 < num_tokens) ? 1'b1 : 1'b0);

assign icmp_ln328_fu_305_p2 = ((hct_reg_190 == 4'd8) ? 1'b1 : 1'b0);

assign t_2_fu_290_p2 = (t_fu_106 + 11'd1);

assign tmp_s_fu_321_p3 = {{trunc_ln328_reg_519}, {7'd0}};

assign trunc_ln328_fu_301_p1 = hct_reg_190[2:0];

assign vq_buf_0_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_0_address0;

assign vq_buf_0_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_0_ce0;

assign vq_buf_1_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_1_address0;

assign vq_buf_1_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_1_ce0;

assign vq_buf_2_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_2_address0;

assign vq_buf_2_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_2_ce0;

assign vq_buf_3_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_3_address0;

assign vq_buf_3_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_3_ce0;

assign vq_buf_4_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_4_address0;

assign vq_buf_4_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_4_ce0;

assign vq_buf_5_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_5_address0;

assign vq_buf_5_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_5_ce0;

assign vq_buf_6_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_6_address0;

assign vq_buf_6_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_6_ce0;

assign vq_buf_7_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_7_address0;

assign vq_buf_7_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vq_buf_7_ce0;

assign vs_buf_address0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vs_buf_address0;

assign vs_buf_ce0 = grp_shared_rv_bmm_head_Pipeline_VITIS_LOOP_336_4_fu_219_vs_buf_ce0;

always @ (posedge ap_clk) begin
    tmp_s_reg_532[6:0] <= 7'b0000000;
end

endmodule //RV_GEMM_shared_rv_bmm_head
