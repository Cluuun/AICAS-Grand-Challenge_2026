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

module SILU_GELU_regslice_both
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

module SILU_GELU_regslice_both_w1
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

module SILU_GELU_pack (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        silu_stream_dout,
        silu_stream_num_data_valid,
        silu_stream_fifo_cap,
        silu_stream_empty_n,
        silu_stream_read,
        silu_stream1_din,
        silu_stream1_num_data_valid,
        silu_stream1_fifo_cap,
        silu_stream1_full_n,
        silu_stream1_write
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [20:0] silu_stream_dout;
input  [2:0] silu_stream_num_data_valid;
input  [2:0] silu_stream_fifo_cap;
input   silu_stream_empty_n;
output   silu_stream_read;
output  [167:0] silu_stream1_din;
input  [4:0] silu_stream1_num_data_valid;
input  [4:0] silu_stream1_fifo_cap;
input   silu_stream1_full_n;
output   silu_stream1_write;

reg ap_idle;
reg silu_stream_read;
reg silu_stream1_write;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [1:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
reg   [0:0] icmp_ln83_reg_450;
reg   [0:0] icmp_ln83_reg_450_pp0_iter2_reg;
reg   [0:0] icmp_ln88_1_reg_476;
reg    ap_predicate_op69_write_state4;
reg    ap_block_state4_pp0_stage0_iter3;
wire    ap_CS_iter3_fsm_state4;
wire   [0:0] icmp_ln83_fu_163_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    silu_stream_blk_n;
reg    silu_stream1_blk_n;
wire   [0:0] icmp_ln83_reg_450_pp0_iter0_reg;
reg   [0:0] icmp_ln83_reg_450_pp0_iter1_reg;
wire   [0:0] icmp_ln84_fu_175_p2;
reg   [0:0] icmp_ln84_reg_454;
wire   [0:0] or_ln84_fu_225_p2;
reg   [0:0] or_ln84_reg_460;
reg   [0:0] or_ln84_reg_460_pp0_iter2_reg;
wire   [3:0] t_2_fu_236_p3;
reg   [3:0] t_2_reg_471;
wire   [0:0] icmp_ln88_1_fu_249_p2;
reg   [3:0] t_fu_54;
wire    ap_loop_init;
reg   [20:0] p_0_0_08_155_fu_58;
wire   [20:0] select_ln84_5_fu_310_p3;
reg   [20:0] p_0_0_08_257_fu_62;
wire   [20:0] select_ln84_4_fu_303_p3;
reg   [20:0] p_0_0_08_359_fu_66;
wire   [20:0] select_ln84_3_fu_296_p3;
reg   [20:0] p_0_0_08_461_fu_70;
wire   [20:0] select_ln84_2_fu_289_p3;
reg   [20:0] p_0_0_08_563_fu_74;
wire   [20:0] select_ln84_1_fu_282_p3;
reg   [20:0] p_0_0_08_665_fu_78;
wire   [20:0] select_ln84_fu_275_p3;
reg   [20:0] p_0_0_0_0_0_067_fu_82;
reg   [12:0] indvar_flatten_fu_86;
wire   [12:0] select_ln84_7_fu_187_p3;
reg   [12:0] ap_sig_allocacmp_indvar_flatten_load;
reg   [14:0] indvar_flatten18_fu_90;
wire   [14:0] add_ln83_fu_169_p2;
reg   [14:0] ap_sig_allocacmp_indvar_flatten18_load;
wire   [12:0] add_ln84_fu_181_p2;
wire   [0:0] icmp_ln88_fu_213_p2;
wire   [0:0] xor_ln83_fu_208_p2;
wire   [0:0] and_ln83_fu_219_p2;
wire   [3:0] add_ln88_fu_230_p2;
wire   [20:0] select_ln84_6_fu_317_p3;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
reg    ap_ST_iter3_fsm_state4_blk;
wire    ap_start_int;
reg    ap_condition_98;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 t_fu_54 = 4'd0;
//#0 p_0_0_08_155_fu_58 = 21'd0;
//#0 p_0_0_08_257_fu_62 = 21'd0;
//#0 p_0_0_08_359_fu_66 = 21'd0;
//#0 p_0_0_08_461_fu_70 = 21'd0;
//#0 p_0_0_08_563_fu_74 = 21'd0;
//#0 p_0_0_08_665_fu_78 = 21'd0;
//#0 p_0_0_0_0_0_067_fu_82 = 21'd0;
//#0 indvar_flatten_fu_86 = 13'd0;
//#0 indvar_flatten18_fu_90 = 15'd0;
//#0 ap_done_reg = 1'b0;
end

SILU_GELU_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter3_fsm_state4) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b1) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter3_fsm_state4) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b0) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_98)) begin
        if ((icmp_ln83_fu_163_p2 == 1'd0)) begin
            indvar_flatten18_fu_90 <= add_ln83_fu_169_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten18_fu_90 <= 15'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_98)) begin
        if ((icmp_ln83_fu_163_p2 == 1'd0)) begin
            indvar_flatten_fu_86 <= select_ln84_7_fu_187_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_86 <= 13'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_08_155_fu_58 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_08_155_fu_58 <= select_ln84_5_fu_310_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_08_257_fu_62 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_08_257_fu_62 <= select_ln84_4_fu_303_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_08_359_fu_66 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_08_359_fu_66 <= select_ln84_3_fu_296_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_08_461_fu_70 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_08_461_fu_70 <= select_ln84_2_fu_289_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_08_563_fu_74 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_08_563_fu_74 <= select_ln84_1_fu_282_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_08_665_fu_78 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_08_665_fu_78 <= select_ln84_fu_275_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_0_0_0_067_fu_82 <= 21'd0;
    end else if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        p_0_0_0_0_0_067_fu_82 <= silu_stream_dout;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        t_fu_54 <= 4'd0;
    end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln83_reg_450_pp0_iter0_reg == 1'd0))) begin
        t_fu_54 <= t_2_fu_236_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln83_reg_450 <= icmp_ln83_fu_163_p2;
        icmp_ln84_reg_454 <= icmp_ln84_fu_175_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln83_reg_450_pp0_iter1_reg <= icmp_ln83_reg_450;
        or_ln84_reg_460 <= or_ln84_fu_225_p2;
        t_2_reg_471 <= t_2_fu_236_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        icmp_ln83_reg_450_pp0_iter2_reg <= icmp_ln83_reg_450_pp0_iter1_reg;
        icmp_ln88_1_reg_476 <= icmp_ln88_1_fu_249_p2;
        or_ln84_reg_460_pp0_iter2_reg <= or_ln84_reg_460;
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

always @ (*) begin
    if ((1'b1 == ap_block_state4_pp0_stage0_iter3)) begin
        ap_ST_iter3_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter3_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (icmp_ln83_fu_163_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter3_fsm_state4) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b1) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten18_load = 15'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten18_load = indvar_flatten18_fu_90;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 13'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_86;
    end
end

always @ (*) begin
    if (((ap_predicate_op69_write_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        silu_stream1_blk_n = silu_stream1_full_n;
    end else begin
        silu_stream1_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_predicate_op69_write_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        silu_stream1_write = 1'b1;
    end else begin
        silu_stream1_write = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        silu_stream_blk_n = silu_stream_empty_n;
    end else begin
        silu_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
        silu_stream_read = 1'b1;
    end else begin
        silu_stream_read = 1'b0;
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
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if (((1'b0 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state4_pp0_stage0_iter3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state4_pp0_stage0_iter3)) | ((icmp_ln83_reg_450_pp0_iter2_reg == 1'd1) & (1'b1 == ap_CS_iter3_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage0_iter3)))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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

assign add_ln83_fu_169_p2 = (ap_sig_allocacmp_indvar_flatten18_load + 15'd1);

assign add_ln84_fu_181_p2 = (ap_sig_allocacmp_indvar_flatten_load + 13'd1);

assign add_ln88_fu_230_p2 = (t_fu_54 + 4'd1);

assign and_ln83_fu_219_p2 = (xor_ln83_fu_208_p2 & icmp_ln88_fu_213_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state4 = ap_CS_iter3_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state4_pp0_stage0_iter3 = (((ap_predicate_op69_write_state4 == 1'b1) & (silu_stream1_full_n == 1'b0)) | ((icmp_ln83_reg_450_pp0_iter2_reg == 1'd0) & (silu_stream_empty_n == 1'b0)));
end

always @ (*) begin
    ap_condition_98 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op69_write_state4 = ((icmp_ln88_1_reg_476 == 1'd1) & (icmp_ln83_reg_450_pp0_iter2_reg == 1'd0));
end

assign icmp_ln83_fu_163_p2 = ((ap_sig_allocacmp_indvar_flatten18_load == 15'd20480) ? 1'b1 : 1'b0);

assign icmp_ln83_reg_450_pp0_iter0_reg = icmp_ln83_reg_450;

assign icmp_ln84_fu_175_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 13'd2560) ? 1'b1 : 1'b0);

assign icmp_ln88_1_fu_249_p2 = ((t_2_reg_471 == 4'd8) ? 1'b1 : 1'b0);

assign icmp_ln88_fu_213_p2 = ((t_fu_54 == 4'd8) ? 1'b1 : 1'b0);

assign or_ln84_fu_225_p2 = (icmp_ln84_reg_454 | and_ln83_fu_219_p2);

assign select_ln84_1_fu_282_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_08_665_fu_78);

assign select_ln84_2_fu_289_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_08_563_fu_74);

assign select_ln84_3_fu_296_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_08_461_fu_70);

assign select_ln84_4_fu_303_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_08_359_fu_66);

assign select_ln84_5_fu_310_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_08_257_fu_62);

assign select_ln84_6_fu_317_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_08_155_fu_58);

assign select_ln84_7_fu_187_p3 = ((icmp_ln84_fu_175_p2[0:0] == 1'b1) ? 13'd1 : add_ln84_fu_181_p2);

assign select_ln84_fu_275_p3 = ((or_ln84_reg_460_pp0_iter2_reg[0:0] == 1'b1) ? 21'd0 : p_0_0_0_0_0_067_fu_82);

assign silu_stream1_din = {{{{{{{{silu_stream_dout}, {select_ln84_fu_275_p3}}, {select_ln84_1_fu_282_p3}}, {select_ln84_2_fu_289_p3}}, {select_ln84_3_fu_296_p3}}, {select_ln84_4_fu_303_p3}}, {select_ln84_5_fu_310_p3}}, {select_ln84_6_fu_317_p3}};

assign t_2_fu_236_p3 = ((or_ln84_fu_225_p2[0:0] == 1'b1) ? 4'd1 : add_ln88_fu_230_p2);

assign xor_ln83_fu_208_p2 = (icmp_ln84_reg_454 ^ 1'd1);

endmodule //SILU_GELU_pack
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module SILU_GELU_do_silu_parallel_SILU_TABLE_ROM_AUTO_1R (
    address0, ce0, q0, 
    reset, clk);

parameter DataWidth = 8;
parameter AddressWidth = 9;
parameter AddressRange = 512;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];


initial begin
     
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_do_silu_parallel_SILU_TABLE_ROM_AUTO_1R.dat", rom0);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
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

module SILU_GELU_flow_control_loop_pipe(
        ap_clk,
        ap_rst,
        ap_start,
        ap_ready,
        ap_done,
        ap_continue,
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
input   ap_continue;

//Block level handshake with loop body
output  ap_start_int;
input   ap_ready_int;
input   ap_done_int;
output  ap_continue_int;

//Init live in variables
output   ap_loop_init;
reg ap_loop_init;

//Exit signal from loop body
input   ap_loop_exit_ready;
input   ap_loop_exit_done;

// power-on initialization
initial begin
//#0 ap_loop_init = 1'b1;
end

assign ap_start_int = ap_start;

assign ap_continue_int = ap_continue;

assign ap_done = ap_loop_exit_done;

assign ap_ready = ap_loop_exit_ready;

//ap_loop_init is valid for the first II
//of the first loop run so as to enable
//the init block ops which are pushed into
//the first state of the pipeline region
always @ (posedge ap_clk)
begin
    if (ap_rst == 1'b1) begin
        ap_loop_init <= 1'b1;
    end else if(ap_loop_exit_ready == 1'b1) begin
        ap_loop_init <= 1'b1;
    end else if(ap_ready_int == 1'b1) begin
        ap_loop_init <= 1'b0;
    end
end

endmodule
        
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_dataflow_in_loop_VITIS_LOOP_236_1_1 (
        mode,
        mlp_i_stream_TDATA,
        q_stream_TDATA,
        s_stream_TDATA,
        ap_clk,
        ap_rst,
        mode_ap_vld,
        mlp_i_stream_TVALID,
        mlp_i_stream_TREADY,
        ap_start,
        q_stream_TVALID,
        q_stream_TREADY,
        s_stream_TVALID,
        s_stream_TREADY,
        ap_done,
        ap_ready,
        ap_idle,
        ap_continue
);


input  [0:0] mode;
input  [223:0] mlp_i_stream_TDATA;
output  [63:0] q_stream_TDATA;
output  [7:0] s_stream_TDATA;
input   ap_clk;
input   ap_rst;
input   mode_ap_vld;
input   mlp_i_stream_TVALID;
output   mlp_i_stream_TREADY;
input   ap_start;
output   q_stream_TVALID;
input   q_stream_TREADY;
output   s_stream_TVALID;
input   s_stream_TREADY;
output   ap_done;
output   ap_ready;
output   ap_idle;
input   ap_continue;

wire    shared_activation_to_common3_U0_ap_start;
wire    shared_activation_to_common3_U0_ap_done;
wire    shared_activation_to_common3_U0_ap_continue;
wire    shared_activation_to_common3_U0_ap_idle;
wire    shared_activation_to_common3_U0_ap_ready;
wire    shared_activation_to_common3_U0_start_out;
wire    shared_activation_to_common3_U0_start_write;
wire    shared_activation_to_common3_U0_mlp_i_stream_TREADY;
wire   [215:0] shared_activation_to_common3_U0_act_stream_din;
wire    shared_activation_to_common3_U0_act_stream_write;
wire   [0:0] shared_activation_to_common3_U0_mode_c_din;
wire    shared_activation_to_common3_U0_mode_c_write;
wire    shared_mlp_quantize4_U0_ap_start;
wire    shared_mlp_quantize4_U0_ap_done;
wire    shared_mlp_quantize4_U0_ap_continue;
wire    shared_mlp_quantize4_U0_ap_idle;
wire    shared_mlp_quantize4_U0_ap_ready;
wire    shared_mlp_quantize4_U0_mode_read;
wire    shared_mlp_quantize4_U0_act_stream_i_read;
wire   [63:0] shared_mlp_quantize4_U0_q_stream_TDATA;
wire    shared_mlp_quantize4_U0_q_stream_TVALID;
wire   [7:0] shared_mlp_quantize4_U0_s_stream_TDATA;
wire    shared_mlp_quantize4_U0_s_stream_TVALID;
wire    act_stream_i_full_n;
wire   [215:0] act_stream_i_dout;
wire   [5:0] act_stream_i_num_data_valid;
wire   [5:0] act_stream_i_fifo_cap;
wire    act_stream_i_empty_n;
wire    mode_c_full_n;
wire   [0:0] mode_c_dout;
wire   [2:0] mode_c_num_data_valid;
wire   [2:0] mode_c_fifo_cap;
wire    mode_c_empty_n;
wire   [0:0] start_for_shared_mlp_quantize4_U0_din;
wire    start_for_shared_mlp_quantize4_U0_full_n;
wire   [0:0] start_for_shared_mlp_quantize4_U0_dout;
wire    start_for_shared_mlp_quantize4_U0_empty_n;

SILU_GELU_shared_activation_to_common3 shared_activation_to_common3_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(shared_activation_to_common3_U0_ap_start),
    .start_full_n(start_for_shared_mlp_quantize4_U0_full_n),
    .ap_done(shared_activation_to_common3_U0_ap_done),
    .ap_continue(shared_activation_to_common3_U0_ap_continue),
    .ap_idle(shared_activation_to_common3_U0_ap_idle),
    .ap_ready(shared_activation_to_common3_U0_ap_ready),
    .start_out(shared_activation_to_common3_U0_start_out),
    .start_write(shared_activation_to_common3_U0_start_write),
    .mode(mode),
    .mlp_i_stream_TDATA(mlp_i_stream_TDATA),
    .mlp_i_stream_TVALID(mlp_i_stream_TVALID),
    .mlp_i_stream_TREADY(shared_activation_to_common3_U0_mlp_i_stream_TREADY),
    .act_stream_din(shared_activation_to_common3_U0_act_stream_din),
    .act_stream_num_data_valid(act_stream_i_num_data_valid),
    .act_stream_fifo_cap(act_stream_i_fifo_cap),
    .act_stream_full_n(act_stream_i_full_n),
    .act_stream_write(shared_activation_to_common3_U0_act_stream_write),
    .mode_c_din(shared_activation_to_common3_U0_mode_c_din),
    .mode_c_num_data_valid(mode_c_num_data_valid),
    .mode_c_fifo_cap(mode_c_fifo_cap),
    .mode_c_full_n(mode_c_full_n),
    .mode_c_write(shared_activation_to_common3_U0_mode_c_write)
);

SILU_GELU_shared_mlp_quantize4 shared_mlp_quantize4_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(shared_mlp_quantize4_U0_ap_start),
    .ap_done(shared_mlp_quantize4_U0_ap_done),
    .ap_continue(shared_mlp_quantize4_U0_ap_continue),
    .ap_idle(shared_mlp_quantize4_U0_ap_idle),
    .ap_ready(shared_mlp_quantize4_U0_ap_ready),
    .mode_dout(mode_c_dout),
    .mode_num_data_valid(mode_c_num_data_valid),
    .mode_fifo_cap(mode_c_fifo_cap),
    .mode_empty_n(mode_c_empty_n),
    .mode_read(shared_mlp_quantize4_U0_mode_read),
    .act_stream_i_dout(act_stream_i_dout),
    .act_stream_i_num_data_valid(act_stream_i_num_data_valid),
    .act_stream_i_fifo_cap(act_stream_i_fifo_cap),
    .act_stream_i_empty_n(act_stream_i_empty_n),
    .act_stream_i_read(shared_mlp_quantize4_U0_act_stream_i_read),
    .q_stream_TDATA(shared_mlp_quantize4_U0_q_stream_TDATA),
    .q_stream_TVALID(shared_mlp_quantize4_U0_q_stream_TVALID),
    .q_stream_TREADY(q_stream_TREADY),
    .s_stream_TDATA(shared_mlp_quantize4_U0_s_stream_TDATA),
    .s_stream_TVALID(shared_mlp_quantize4_U0_s_stream_TVALID),
    .s_stream_TREADY(s_stream_TREADY)
);

SILU_GELU_fifo_w216_d32_A act_stream_i_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(shared_activation_to_common3_U0_act_stream_din),
    .if_full_n(act_stream_i_full_n),
    .if_write(shared_activation_to_common3_U0_act_stream_write),
    .if_dout(act_stream_i_dout),
    .if_num_data_valid(act_stream_i_num_data_valid),
    .if_fifo_cap(act_stream_i_fifo_cap),
    .if_empty_n(act_stream_i_empty_n),
    .if_read(shared_mlp_quantize4_U0_act_stream_i_read)
);

SILU_GELU_fifo_w1_d2_S mode_c_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(shared_activation_to_common3_U0_mode_c_din),
    .if_full_n(mode_c_full_n),
    .if_write(shared_activation_to_common3_U0_mode_c_write),
    .if_dout(mode_c_dout),
    .if_num_data_valid(mode_c_num_data_valid),
    .if_fifo_cap(mode_c_fifo_cap),
    .if_empty_n(mode_c_empty_n),
    .if_read(shared_mlp_quantize4_U0_mode_read)
);

SILU_GELU_start_for_shared_mlp_quantize4_U0 start_for_shared_mlp_quantize4_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_shared_mlp_quantize4_U0_din),
    .if_full_n(start_for_shared_mlp_quantize4_U0_full_n),
    .if_write(shared_activation_to_common3_U0_start_write),
    .if_dout(start_for_shared_mlp_quantize4_U0_dout),
    .if_empty_n(start_for_shared_mlp_quantize4_U0_empty_n),
    .if_read(shared_mlp_quantize4_U0_ap_ready)
);

assign ap_done = shared_mlp_quantize4_U0_ap_done;

assign ap_idle = (shared_mlp_quantize4_U0_ap_idle & shared_activation_to_common3_U0_ap_idle);

assign ap_ready = shared_activation_to_common3_U0_ap_ready;

assign mlp_i_stream_TREADY = shared_activation_to_common3_U0_mlp_i_stream_TREADY;

assign q_stream_TDATA = shared_mlp_quantize4_U0_q_stream_TDATA;

assign q_stream_TVALID = shared_mlp_quantize4_U0_q_stream_TVALID;

assign s_stream_TDATA = shared_mlp_quantize4_U0_s_stream_TDATA;

assign s_stream_TVALID = shared_mlp_quantize4_U0_s_stream_TVALID;

assign shared_activation_to_common3_U0_ap_continue = 1'b1;

assign shared_activation_to_common3_U0_ap_start = ap_start;

assign shared_mlp_quantize4_U0_ap_continue = ap_continue;

assign shared_mlp_quantize4_U0_ap_start = start_for_shared_mlp_quantize4_U0_empty_n;

assign start_for_shared_mlp_quantize4_U0_din = 1'b1;

endmodule //SILU_GELU_dataflow_in_loop_VITIS_LOOP_236_1_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module SILU_GELU_vit_gelu_to_common_GELU_TABLE_ROM_AUTO_1R (
    address0, ce0, q0, 
    address1, ce1, q1, 
    address2, ce2, q2, 
    address3, ce3, q3, 
    address4, ce4, q4, 
    address5, ce5, q5, 
    address6, ce6, q6, 
    address7, ce7, q7, 
    reset, clk);

parameter DataWidth = 8;
parameter AddressWidth = 12;
parameter AddressRange = 4096;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;
 
input[AddressWidth-1:0] address1;
input ce1;
output reg[DataWidth-1:0] q1;
 
input[AddressWidth-1:0] address2;
input ce2;
output reg[DataWidth-1:0] q2;
 
input[AddressWidth-1:0] address3;
input ce3;
output reg[DataWidth-1:0] q3;
 
input[AddressWidth-1:0] address4;
input ce4;
output reg[DataWidth-1:0] q4;
 
input[AddressWidth-1:0] address5;
input ce5;
output reg[DataWidth-1:0] q5;
 
input[AddressWidth-1:0] address6;
input ce6;
output reg[DataWidth-1:0] q6;
 
input[AddressWidth-1:0] address7;
input ce7;
output reg[DataWidth-1:0] q7;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];
 
reg [DataWidth-1:0] rom1[0:AddressRange-1];
 
reg [DataWidth-1:0] rom2[0:AddressRange-1];
 
reg [DataWidth-1:0] rom3[0:AddressRange-1];


initial begin
     
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_ROM_AUTO_1R.dat", rom0); 
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_ROM_AUTO_1R.dat", rom1); 
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_ROM_AUTO_1R.dat", rom2); 
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_ROM_AUTO_1R.dat", rom3);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
    end
end
  
always @(posedge clk) 
begin 
    if (ce1) 
    begin
        q1 <= rom0[address1];
    end
end
  
always @(posedge clk) 
begin 
    if (ce2) 
    begin
        q2 <= rom1[address2];
    end
end
  
always @(posedge clk) 
begin 
    if (ce3) 
    begin
        q3 <= rom1[address3];
    end
end
  
always @(posedge clk) 
begin 
    if (ce4) 
    begin
        q4 <= rom2[address4];
    end
end
  
always @(posedge clk) 
begin 
    if (ce5) 
    begin
        q5 <= rom2[address5];
    end
end
  
always @(posedge clk) 
begin 
    if (ce6) 
    begin
        q6 <= rom3[address6];
    end
end
  
always @(posedge clk) 
begin 
    if (ce7) 
    begin
        q7 <= rom3[address7];
    end
end


endmodule

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_vit_gelu_to_common (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        mlp_i_stream_TVALID,
        act_stream_din,
        act_stream_num_data_valid,
        act_stream_fifo_cap,
        act_stream_full_n,
        act_stream_write,
        mlp_i_stream_TDATA,
        mlp_i_stream_TREADY
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter4_fsm_state5 = 2'd2;
parameter    ap_ST_iter5_fsm_state6 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   mlp_i_stream_TVALID;
output  [215:0] act_stream_din;
input  [5:0] act_stream_num_data_valid;
input  [5:0] act_stream_fifo_cap;
input   act_stream_full_n;
output   act_stream_write;
input  [223:0] mlp_i_stream_TDATA;
output   mlp_i_stream_TREADY;

reg ap_idle;
reg act_stream_write;
reg mlp_i_stream_TREADY;

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
wire   [0:0] icmp_ln141_fu_327_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
reg   [0:0] icmp_ln141_reg_1382;
reg   [0:0] icmp_ln141_reg_1382_pp0_iter4_reg;
reg    ap_block_state6_pp0_stage0_iter5;
wire    ap_CS_iter5_fsm_state6;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
wire   [11:0] GELU_TABLE_address0;
reg    GELU_TABLE_ce0;
wire   [7:0] GELU_TABLE_q0;
wire   [11:0] GELU_TABLE_address1;
reg    GELU_TABLE_ce1;
wire   [7:0] GELU_TABLE_q1;
wire   [11:0] GELU_TABLE_address2;
reg    GELU_TABLE_ce2;
wire   [7:0] GELU_TABLE_q2;
wire   [11:0] GELU_TABLE_address3;
reg    GELU_TABLE_ce3;
wire   [7:0] GELU_TABLE_q3;
wire   [11:0] GELU_TABLE_address4;
reg    GELU_TABLE_ce4;
wire   [7:0] GELU_TABLE_q4;
wire   [11:0] GELU_TABLE_address5;
reg    GELU_TABLE_ce5;
wire   [7:0] GELU_TABLE_q5;
wire   [11:0] GELU_TABLE_address6;
reg    GELU_TABLE_ce6;
wire   [7:0] GELU_TABLE_q6;
wire   [11:0] GELU_TABLE_address7;
reg    GELU_TABLE_ce7;
wire   [7:0] GELU_TABLE_q7;
wire   [4:0] GELU_TABLE_S_address0;
reg    GELU_TABLE_S_ce0;
wire   [3:0] GELU_TABLE_S_q0;
wire   [4:0] GELU_TABLE_S_address1;
reg    GELU_TABLE_S_ce1;
wire   [3:0] GELU_TABLE_S_q1;
wire   [4:0] GELU_TABLE_S_address2;
reg    GELU_TABLE_S_ce2;
wire   [3:0] GELU_TABLE_S_q2;
wire   [4:0] GELU_TABLE_S_address3;
reg    GELU_TABLE_S_ce3;
wire   [3:0] GELU_TABLE_S_q3;
wire   [4:0] GELU_TABLE_S_address4;
reg    GELU_TABLE_S_ce4;
wire   [3:0] GELU_TABLE_S_q4;
wire   [4:0] GELU_TABLE_S_address5;
reg    GELU_TABLE_S_ce5;
wire   [3:0] GELU_TABLE_S_q5;
wire   [4:0] GELU_TABLE_S_address6;
reg    GELU_TABLE_S_ce6;
wire   [3:0] GELU_TABLE_S_q6;
wire   [4:0] GELU_TABLE_S_address7;
reg    GELU_TABLE_S_ce7;
wire   [3:0] GELU_TABLE_S_q7;
reg    mlp_i_stream_TDATA_blk_n;
reg    act_stream_blk_n;
wire   [0:0] icmp_ln141_reg_1382_pp0_iter0_reg;
reg   [0:0] icmp_ln141_reg_1382_pp0_iter1_reg;
reg   [0:0] icmp_ln141_reg_1382_pp0_iter2_reg;
reg   [0:0] icmp_ln141_reg_1382_pp0_iter3_reg;
wire   [27:0] value_fu_339_p1;
reg   [27:0] value_reg_1386;
reg   [27:0] x_5_reg_1391;
reg   [27:0] x_7_reg_1396;
reg   [27:0] x_9_reg_1401;
reg   [27:0] x_11_reg_1406;
reg   [27:0] x_12_reg_1411;
reg   [27:0] x_13_reg_1416;
reg   [27:0] x_14_reg_1421;
wire   [28:0] add_ln126_fu_421_p2;
reg   [28:0] add_ln126_reg_1426;
reg   [0:0] tmp_1_reg_1432;
wire   [0:0] icmp_ln43_fu_445_p2;
reg   [0:0] icmp_ln43_reg_1438;
wire   [28:0] add_ln126_1_fu_454_p2;
reg   [28:0] add_ln126_1_reg_1443;
reg   [0:0] tmp_3_reg_1449;
wire   [0:0] icmp_ln43_1_fu_478_p2;
reg   [0:0] icmp_ln43_1_reg_1455;
wire   [28:0] add_ln126_2_fu_487_p2;
reg   [28:0] add_ln126_2_reg_1460;
reg   [0:0] tmp_5_reg_1466;
wire   [0:0] icmp_ln43_2_fu_511_p2;
reg   [0:0] icmp_ln43_2_reg_1472;
wire   [28:0] add_ln126_3_fu_520_p2;
reg   [28:0] add_ln126_3_reg_1477;
reg   [0:0] tmp_7_reg_1483;
wire   [0:0] icmp_ln43_3_fu_544_p2;
reg   [0:0] icmp_ln43_3_reg_1489;
wire   [28:0] add_ln126_4_fu_553_p2;
reg   [28:0] add_ln126_4_reg_1494;
reg   [0:0] tmp_9_reg_1500;
wire   [0:0] icmp_ln43_4_fu_577_p2;
reg   [0:0] icmp_ln43_4_reg_1506;
wire   [28:0] add_ln126_5_fu_586_p2;
reg   [28:0] add_ln126_5_reg_1511;
reg   [0:0] tmp_11_reg_1517;
wire   [0:0] icmp_ln43_5_fu_610_p2;
reg   [0:0] icmp_ln43_5_reg_1523;
wire   [28:0] add_ln126_6_fu_619_p2;
reg   [28:0] add_ln126_6_reg_1528;
reg   [0:0] tmp_13_reg_1534;
wire   [0:0] icmp_ln43_6_fu_643_p2;
reg   [0:0] icmp_ln43_6_reg_1540;
wire   [28:0] add_ln126_7_fu_652_p2;
reg   [28:0] add_ln126_7_reg_1545;
reg   [0:0] tmp_15_reg_1551;
wire   [0:0] icmp_ln43_7_fu_676_p2;
reg   [0:0] icmp_ln43_7_reg_1557;
wire   [11:0] lut_idx_fu_717_p3;
reg   [11:0] lut_idx_reg_1562;
wire   [11:0] lut_idx_1_fu_781_p3;
reg   [11:0] lut_idx_1_reg_1572;
wire   [11:0] lut_idx_2_fu_845_p3;
reg   [11:0] lut_idx_2_reg_1582;
wire   [11:0] lut_idx_3_fu_909_p3;
reg   [11:0] lut_idx_3_reg_1592;
wire   [11:0] lut_idx_4_fu_973_p3;
reg   [11:0] lut_idx_4_reg_1602;
wire   [11:0] lut_idx_5_fu_1037_p3;
reg   [11:0] lut_idx_5_reg_1612;
wire   [11:0] lut_idx_6_fu_1101_p3;
reg   [11:0] lut_idx_6_reg_1622;
wire   [11:0] lut_idx_7_fu_1165_p3;
reg   [11:0] lut_idx_7_reg_1632;
reg   [3:0] GELU_TABLE_S_load_reg_1647;
reg   [3:0] GELU_TABLE_S_load_reg_1647_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_1_reg_1657;
reg   [3:0] GELU_TABLE_S_load_1_reg_1657_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_2_reg_1667;
reg   [3:0] GELU_TABLE_S_load_2_reg_1667_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_3_reg_1677;
reg   [3:0] GELU_TABLE_S_load_3_reg_1677_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_4_reg_1687;
reg   [3:0] GELU_TABLE_S_load_4_reg_1687_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_5_reg_1697;
reg   [3:0] GELU_TABLE_S_load_5_reg_1697_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_6_reg_1707;
reg   [3:0] GELU_TABLE_S_load_6_reg_1707_pp0_iter4_reg;
reg   [3:0] GELU_TABLE_S_load_7_reg_1717;
reg   [3:0] GELU_TABLE_S_load_7_reg_1717_pp0_iter4_reg;
reg   [7:0] GELU_TABLE_load_reg_1722;
reg   [7:0] GELU_TABLE_load_1_reg_1727;
reg   [7:0] GELU_TABLE_load_2_reg_1732;
reg   [7:0] GELU_TABLE_load_3_reg_1737;
reg   [7:0] GELU_TABLE_load_4_reg_1742;
reg   [7:0] GELU_TABLE_load_5_reg_1747;
reg   [7:0] GELU_TABLE_load_6_reg_1752;
reg   [7:0] GELU_TABLE_load_7_reg_1757;
wire   [63:0] zext_ln131_1_fu_741_p1;
wire   [63:0] zext_ln131_3_fu_805_p1;
wire   [63:0] zext_ln131_5_fu_869_p1;
wire   [63:0] zext_ln131_7_fu_933_p1;
wire   [63:0] zext_ln131_9_fu_997_p1;
wire   [63:0] zext_ln131_11_fu_1061_p1;
wire   [63:0] zext_ln131_13_fu_1125_p1;
wire   [63:0] zext_ln131_15_fu_1189_p1;
wire   [63:0] zext_ln131_fu_1194_p1;
wire   [63:0] zext_ln131_2_fu_1198_p1;
wire   [63:0] zext_ln131_4_fu_1202_p1;
wire   [63:0] zext_ln131_6_fu_1206_p1;
wire   [63:0] zext_ln131_8_fu_1210_p1;
wire   [63:0] zext_ln131_10_fu_1214_p1;
wire   [63:0] zext_ln131_12_fu_1218_p1;
wire   [63:0] zext_ln131_14_fu_1222_p1;
reg   [18:0] vec_fu_108;
wire   [18:0] vec_2_fu_333_p2;
wire    ap_loop_init;
reg   [18:0] ap_sig_allocacmp_vec_1;
wire  signed [28:0] sext_ln126_fu_418_p1;
wire   [2:0] tmp_2_fu_435_p4;
wire  signed [28:0] sext_ln126_1_fu_451_p1;
wire   [2:0] tmp_4_fu_468_p4;
wire  signed [28:0] sext_ln126_2_fu_484_p1;
wire   [2:0] tmp_6_fu_501_p4;
wire  signed [28:0] sext_ln126_3_fu_517_p1;
wire   [2:0] tmp_8_fu_534_p4;
wire  signed [28:0] sext_ln126_4_fu_550_p1;
wire   [2:0] tmp_10_fu_567_p4;
wire  signed [28:0] sext_ln126_5_fu_583_p1;
wire   [2:0] tmp_12_fu_600_p4;
wire  signed [28:0] sext_ln126_6_fu_616_p1;
wire   [2:0] tmp_14_fu_633_p4;
wire  signed [28:0] sext_ln126_7_fu_649_p1;
wire   [2:0] tmp_16_fu_666_p4;
wire   [0:0] xor_ln42_fu_700_p2;
wire   [0:0] or_ln42_fu_713_p2;
wire   [11:0] select_ln42_fu_705_p3;
wire   [11:0] trunc_ln126_2_fu_682_p4;
wire   [4:0] select_ln42_2_fu_725_p3;
wire   [4:0] trunc_ln127_2_fu_691_p4;
wire   [4:0] lut_s_idx_fu_733_p3;
wire   [0:0] xor_ln42_1_fu_764_p2;
wire   [0:0] or_ln42_1_fu_777_p2;
wire   [11:0] select_ln42_4_fu_769_p3;
wire   [11:0] trunc_ln_fu_746_p4;
wire   [4:0] select_ln42_6_fu_789_p3;
wire   [4:0] trunc_ln1_fu_755_p4;
wire   [4:0] lut_s_idx_1_fu_797_p3;
wire   [0:0] xor_ln42_2_fu_828_p2;
wire   [0:0] or_ln42_2_fu_841_p2;
wire   [11:0] select_ln42_8_fu_833_p3;
wire   [11:0] trunc_ln126_5_fu_810_p4;
wire   [4:0] select_ln42_10_fu_853_p3;
wire   [4:0] trunc_ln127_5_fu_819_p4;
wire   [4:0] lut_s_idx_2_fu_861_p3;
wire   [0:0] xor_ln42_3_fu_892_p2;
wire   [0:0] or_ln42_3_fu_905_p2;
wire   [11:0] select_ln42_12_fu_897_p3;
wire   [11:0] trunc_ln126_7_fu_874_p4;
wire   [4:0] select_ln42_14_fu_917_p3;
wire   [4:0] trunc_ln127_7_fu_883_p4;
wire   [4:0] lut_s_idx_3_fu_925_p3;
wire   [0:0] xor_ln42_4_fu_956_p2;
wire   [0:0] or_ln42_4_fu_969_p2;
wire   [11:0] select_ln42_16_fu_961_p3;
wire   [11:0] trunc_ln126_9_fu_938_p4;
wire   [4:0] select_ln42_18_fu_981_p3;
wire   [4:0] trunc_ln127_9_fu_947_p4;
wire   [4:0] lut_s_idx_4_fu_989_p3;
wire   [0:0] xor_ln42_5_fu_1020_p2;
wire   [0:0] or_ln42_5_fu_1033_p2;
wire   [11:0] select_ln42_20_fu_1025_p3;
wire   [11:0] trunc_ln126_s_fu_1002_p4;
wire   [4:0] select_ln42_22_fu_1045_p3;
wire   [4:0] trunc_ln127_s_fu_1011_p4;
wire   [4:0] lut_s_idx_5_fu_1053_p3;
wire   [0:0] xor_ln42_6_fu_1084_p2;
wire   [0:0] or_ln42_6_fu_1097_p2;
wire   [11:0] select_ln42_24_fu_1089_p3;
wire   [11:0] trunc_ln126_1_fu_1066_p4;
wire   [4:0] select_ln42_26_fu_1109_p3;
wire   [4:0] trunc_ln127_1_fu_1075_p4;
wire   [4:0] lut_s_idx_6_fu_1117_p3;
wire   [0:0] xor_ln42_7_fu_1148_p2;
wire   [0:0] or_ln42_7_fu_1161_p2;
wire   [11:0] select_ln42_28_fu_1153_p3;
wire   [11:0] trunc_ln126_3_fu_1130_p4;
wire   [4:0] select_ln42_30_fu_1173_p3;
wire   [4:0] trunc_ln127_3_fu_1139_p4;
wire   [4:0] lut_s_idx_7_fu_1181_p3;
wire  signed [21:0] GELU_TABLE_load_cast_fu_1226_p1;
wire   [21:0] GELU_TABLE_S_load_cast_fu_1229_p1;
wire   [21:0] gelu_val_fu_1232_p2;
wire  signed [21:0] GELU_TABLE_load_1_cast_fu_1242_p1;
wire   [21:0] GELU_TABLE_S_load_1_cast_fu_1245_p1;
wire  signed [21:0] GELU_TABLE_load_2_cast_fu_1254_p1;
wire   [21:0] GELU_TABLE_S_load_2_cast_fu_1257_p1;
wire  signed [21:0] GELU_TABLE_load_3_cast_fu_1266_p1;
wire   [21:0] GELU_TABLE_S_load_3_cast_fu_1269_p1;
wire  signed [21:0] GELU_TABLE_load_4_cast_fu_1278_p1;
wire   [21:0] GELU_TABLE_S_load_4_cast_fu_1281_p1;
wire  signed [21:0] GELU_TABLE_load_5_cast_fu_1290_p1;
wire   [21:0] GELU_TABLE_S_load_5_cast_fu_1293_p1;
wire  signed [21:0] GELU_TABLE_load_6_cast_fu_1302_p1;
wire   [21:0] GELU_TABLE_S_load_6_cast_fu_1305_p1;
wire  signed [21:0] GELU_TABLE_load_7_cast_fu_1314_p1;
wire   [21:0] GELU_TABLE_S_load_7_cast_fu_1317_p1;
wire   [21:0] gelu_val_1_fu_1248_p2;
wire   [21:0] gelu_val_2_fu_1260_p2;
wire   [21:0] gelu_val_3_fu_1272_p2;
wire   [21:0] gelu_val_4_fu_1284_p2;
wire   [21:0] gelu_val_5_fu_1296_p2;
wire   [21:0] gelu_val_6_fu_1308_p2;
wire   [21:0] gelu_val_7_fu_1320_p2;
wire  signed [26:0] sext_ln150_5_fu_1346_p1;
wire  signed [26:0] sext_ln150_4_fu_1342_p1;
wire  signed [26:0] sext_ln150_3_fu_1338_p1;
wire  signed [26:0] sext_ln150_2_fu_1334_p1;
wire  signed [26:0] sext_ln150_1_fu_1330_p1;
wire  signed [26:0] sext_ln150_fu_1326_p1;
wire  signed [26:0] sext_ln132_fu_1238_p1;
wire   [210:0] tmp_fu_1350_p9;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
reg    ap_ST_iter5_fsm_state6_blk;
wire    ap_start_int;
reg    ap_condition_117;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 ap_CS_iter4_fsm = 2'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 vec_fu_108 = 19'd0;
//#0 ap_done_reg = 1'b0;
end

SILU_GELU_vit_gelu_to_common_GELU_TABLE_ROM_AUTO_1R #(
    .DataWidth( 8 ),
    .AddressRange( 4096 ),
    .AddressWidth( 12 ))
GELU_TABLE_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(GELU_TABLE_address0),
    .ce0(GELU_TABLE_ce0),
    .q0(GELU_TABLE_q0),
    .address1(GELU_TABLE_address1),
    .ce1(GELU_TABLE_ce1),
    .q1(GELU_TABLE_q1),
    .address2(GELU_TABLE_address2),
    .ce2(GELU_TABLE_ce2),
    .q2(GELU_TABLE_q2),
    .address3(GELU_TABLE_address3),
    .ce3(GELU_TABLE_ce3),
    .q3(GELU_TABLE_q3),
    .address4(GELU_TABLE_address4),
    .ce4(GELU_TABLE_ce4),
    .q4(GELU_TABLE_q4),
    .address5(GELU_TABLE_address5),
    .ce5(GELU_TABLE_ce5),
    .q5(GELU_TABLE_q5),
    .address6(GELU_TABLE_address6),
    .ce6(GELU_TABLE_ce6),
    .q6(GELU_TABLE_q6),
    .address7(GELU_TABLE_address7),
    .ce7(GELU_TABLE_ce7),
    .q7(GELU_TABLE_q7)
);

SILU_GELU_vit_gelu_to_common_GELU_TABLE_S_ROM_AUTO_1R #(
    .DataWidth( 4 ),
    .AddressRange( 32 ),
    .AddressWidth( 5 ))
GELU_TABLE_S_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(GELU_TABLE_S_address0),
    .ce0(GELU_TABLE_S_ce0),
    .q0(GELU_TABLE_S_q0),
    .address1(GELU_TABLE_S_address1),
    .ce1(GELU_TABLE_S_ce1),
    .q1(GELU_TABLE_S_q1),
    .address2(GELU_TABLE_S_address2),
    .ce2(GELU_TABLE_S_ce2),
    .q2(GELU_TABLE_S_q2),
    .address3(GELU_TABLE_S_address3),
    .ce3(GELU_TABLE_S_ce3),
    .q3(GELU_TABLE_S_q3),
    .address4(GELU_TABLE_S_address4),
    .ce4(GELU_TABLE_S_ce4),
    .q4(GELU_TABLE_S_q4),
    .address5(GELU_TABLE_S_address5),
    .ce5(GELU_TABLE_S_ce5),
    .q5(GELU_TABLE_S_q5),
    .address6(GELU_TABLE_S_address6),
    .ce6(GELU_TABLE_S_ce6),
    .q6(GELU_TABLE_S_q6),
    .address7(GELU_TABLE_S_address7),
    .ce7(GELU_TABLE_S_ce7),
    .q7(GELU_TABLE_S_q7)
);

SILU_GELU_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter5_fsm_state6) & (1'b0 == ap_block_state6_pp0_stage0_iter5) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter5_fsm_state6) & (1'b0 == ap_block_state6_pp0_stage0_iter5) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_117)) begin
        if ((icmp_ln141_fu_327_p2 == 1'd0)) begin
            vec_fu_108 <= vec_2_fu_333_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            vec_fu_108 <= 19'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_S_load_1_reg_1657 <= GELU_TABLE_S_q6;
        GELU_TABLE_S_load_2_reg_1667 <= GELU_TABLE_S_q5;
        GELU_TABLE_S_load_3_reg_1677 <= GELU_TABLE_S_q4;
        GELU_TABLE_S_load_4_reg_1687 <= GELU_TABLE_S_q3;
        GELU_TABLE_S_load_5_reg_1697 <= GELU_TABLE_S_q2;
        GELU_TABLE_S_load_6_reg_1707 <= GELU_TABLE_S_q1;
        GELU_TABLE_S_load_7_reg_1717 <= GELU_TABLE_S_q0;
        GELU_TABLE_S_load_reg_1647 <= GELU_TABLE_S_q7;
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln141_reg_1382_pp0_iter3_reg <= icmp_ln141_reg_1382_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        GELU_TABLE_S_load_1_reg_1657_pp0_iter4_reg <= GELU_TABLE_S_load_1_reg_1657;
        GELU_TABLE_S_load_2_reg_1667_pp0_iter4_reg <= GELU_TABLE_S_load_2_reg_1667;
        GELU_TABLE_S_load_3_reg_1677_pp0_iter4_reg <= GELU_TABLE_S_load_3_reg_1677;
        GELU_TABLE_S_load_4_reg_1687_pp0_iter4_reg <= GELU_TABLE_S_load_4_reg_1687;
        GELU_TABLE_S_load_5_reg_1697_pp0_iter4_reg <= GELU_TABLE_S_load_5_reg_1697;
        GELU_TABLE_S_load_6_reg_1707_pp0_iter4_reg <= GELU_TABLE_S_load_6_reg_1707;
        GELU_TABLE_S_load_7_reg_1717_pp0_iter4_reg <= GELU_TABLE_S_load_7_reg_1717;
        GELU_TABLE_S_load_reg_1647_pp0_iter4_reg <= GELU_TABLE_S_load_reg_1647;
        GELU_TABLE_load_1_reg_1727 <= GELU_TABLE_q6;
        GELU_TABLE_load_2_reg_1732 <= GELU_TABLE_q5;
        GELU_TABLE_load_3_reg_1737 <= GELU_TABLE_q4;
        GELU_TABLE_load_4_reg_1742 <= GELU_TABLE_q3;
        GELU_TABLE_load_5_reg_1747 <= GELU_TABLE_q2;
        GELU_TABLE_load_6_reg_1752 <= GELU_TABLE_q1;
        GELU_TABLE_load_7_reg_1757 <= GELU_TABLE_q0;
        GELU_TABLE_load_reg_1722 <= GELU_TABLE_q7;
        icmp_ln141_reg_1382_pp0_iter4_reg <= icmp_ln141_reg_1382_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        add_ln126_1_reg_1443 <= add_ln126_1_fu_454_p2;
        add_ln126_2_reg_1460 <= add_ln126_2_fu_487_p2;
        add_ln126_3_reg_1477 <= add_ln126_3_fu_520_p2;
        add_ln126_4_reg_1494 <= add_ln126_4_fu_553_p2;
        add_ln126_5_reg_1511 <= add_ln126_5_fu_586_p2;
        add_ln126_6_reg_1528 <= add_ln126_6_fu_619_p2;
        add_ln126_7_reg_1545 <= add_ln126_7_fu_652_p2;
        add_ln126_reg_1426 <= add_ln126_fu_421_p2;
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln141_reg_1382_pp0_iter1_reg <= icmp_ln141_reg_1382;
        icmp_ln43_1_reg_1455 <= icmp_ln43_1_fu_478_p2;
        icmp_ln43_2_reg_1472 <= icmp_ln43_2_fu_511_p2;
        icmp_ln43_3_reg_1489 <= icmp_ln43_3_fu_544_p2;
        icmp_ln43_4_reg_1506 <= icmp_ln43_4_fu_577_p2;
        icmp_ln43_5_reg_1523 <= icmp_ln43_5_fu_610_p2;
        icmp_ln43_6_reg_1540 <= icmp_ln43_6_fu_643_p2;
        icmp_ln43_7_reg_1557 <= icmp_ln43_7_fu_676_p2;
        icmp_ln43_reg_1438 <= icmp_ln43_fu_445_p2;
        tmp_11_reg_1517 <= add_ln126_5_fu_586_p2[32'd28];
        tmp_13_reg_1534 <= add_ln126_6_fu_619_p2[32'd28];
        tmp_15_reg_1551 <= add_ln126_7_fu_652_p2[32'd28];
        tmp_1_reg_1432 <= add_ln126_fu_421_p2[32'd28];
        tmp_3_reg_1449 <= add_ln126_1_fu_454_p2[32'd28];
        tmp_5_reg_1466 <= add_ln126_2_fu_487_p2[32'd28];
        tmp_7_reg_1483 <= add_ln126_3_fu_520_p2[32'd28];
        tmp_9_reg_1500 <= add_ln126_4_fu_553_p2[32'd28];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln141_reg_1382 <= icmp_ln141_fu_327_p2;
        value_reg_1386 <= value_fu_339_p1;
        x_11_reg_1406 <= {{mlp_i_stream_TDATA[139:112]}};
        x_12_reg_1411 <= {{mlp_i_stream_TDATA[167:140]}};
        x_13_reg_1416 <= {{mlp_i_stream_TDATA[195:168]}};
        x_14_reg_1421 <= {{mlp_i_stream_TDATA[223:196]}};
        x_5_reg_1391 <= {{mlp_i_stream_TDATA[55:28]}};
        x_7_reg_1396 <= {{mlp_i_stream_TDATA[83:56]}};
        x_9_reg_1401 <= {{mlp_i_stream_TDATA[111:84]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln141_reg_1382_pp0_iter2_reg <= icmp_ln141_reg_1382_pp0_iter1_reg;
        lut_idx_1_reg_1572 <= lut_idx_1_fu_781_p3;
        lut_idx_2_reg_1582 <= lut_idx_2_fu_845_p3;
        lut_idx_3_reg_1592 <= lut_idx_3_fu_909_p3;
        lut_idx_4_reg_1602 <= lut_idx_4_fu_973_p3;
        lut_idx_5_reg_1612 <= lut_idx_5_fu_1037_p3;
        lut_idx_6_reg_1622 <= lut_idx_6_fu_1101_p3;
        lut_idx_7_reg_1632 <= lut_idx_7_fu_1165_p3;
        lut_idx_reg_1562 <= lut_idx_fu_717_p3;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce0 = 1'b1;
    end else begin
        GELU_TABLE_S_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce1 = 1'b1;
    end else begin
        GELU_TABLE_S_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce2 = 1'b1;
    end else begin
        GELU_TABLE_S_ce2 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce3 = 1'b1;
    end else begin
        GELU_TABLE_S_ce3 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce4 = 1'b1;
    end else begin
        GELU_TABLE_S_ce4 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce5 = 1'b1;
    end else begin
        GELU_TABLE_S_ce5 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce6 = 1'b1;
    end else begin
        GELU_TABLE_S_ce6 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        GELU_TABLE_S_ce7 = 1'b1;
    end else begin
        GELU_TABLE_S_ce7 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce0 = 1'b1;
    end else begin
        GELU_TABLE_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce1 = 1'b1;
    end else begin
        GELU_TABLE_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce2 = 1'b1;
    end else begin
        GELU_TABLE_ce2 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce3 = 1'b1;
    end else begin
        GELU_TABLE_ce3 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce4 = 1'b1;
    end else begin
        GELU_TABLE_ce4 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce5 = 1'b1;
    end else begin
        GELU_TABLE_ce5 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce6 = 1'b1;
    end else begin
        GELU_TABLE_ce6 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        GELU_TABLE_ce7 = 1'b1;
    end else begin
        GELU_TABLE_ce7 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln141_reg_1382_pp0_iter4_reg == 1'd0))) begin
        act_stream_blk_n = act_stream_full_n;
    end else begin
        act_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state6) & (1'b0 == ap_block_state6_pp0_stage0_iter5) & (icmp_ln141_reg_1382_pp0_iter4_reg == 1'd0))) begin
        act_stream_write = 1'b1;
    end else begin
        act_stream_write = 1'b0;
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
    if ((1'b1 == ap_block_state6_pp0_stage0_iter5)) begin
        ap_ST_iter5_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter5_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln141_fu_327_p2 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state6) & (1'b0 == ap_block_state6_pp0_stage0_iter5) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1))) begin
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_vec_1 = 19'd0;
    end else begin
        ap_sig_allocacmp_vec_1 = vec_fu_108;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln141_fu_327_p2 == 1'd0) & (ap_start_int == 1'b1))) begin
        mlp_i_stream_TDATA_blk_n = mlp_i_stream_TVALID;
    end else begin
        mlp_i_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln141_fu_327_p2 == 1'd0))) begin
        mlp_i_stream_TREADY = 1'b1;
    end else begin
        mlp_i_stream_TREADY = 1'b0;
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
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
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
            if (((1'b0 == ap_CS_iter4_fsm_state5) & (1'b0 == ap_block_state6_pp0_stage0_iter5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter4_fsm_state5) & (1'b0 == ap_block_state6_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b0 == ap_block_state6_pp0_stage0_iter5) & (icmp_ln141_reg_1382_pp0_iter4_reg == 1'd1)))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
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

assign GELU_TABLE_S_address0 = zext_ln131_15_fu_1189_p1;

assign GELU_TABLE_S_address1 = zext_ln131_13_fu_1125_p1;

assign GELU_TABLE_S_address2 = zext_ln131_11_fu_1061_p1;

assign GELU_TABLE_S_address3 = zext_ln131_9_fu_997_p1;

assign GELU_TABLE_S_address4 = zext_ln131_7_fu_933_p1;

assign GELU_TABLE_S_address5 = zext_ln131_5_fu_869_p1;

assign GELU_TABLE_S_address6 = zext_ln131_3_fu_805_p1;

assign GELU_TABLE_S_address7 = zext_ln131_1_fu_741_p1;

assign GELU_TABLE_S_load_1_cast_fu_1245_p1 = GELU_TABLE_S_load_1_reg_1657_pp0_iter4_reg;

assign GELU_TABLE_S_load_2_cast_fu_1257_p1 = GELU_TABLE_S_load_2_reg_1667_pp0_iter4_reg;

assign GELU_TABLE_S_load_3_cast_fu_1269_p1 = GELU_TABLE_S_load_3_reg_1677_pp0_iter4_reg;

assign GELU_TABLE_S_load_4_cast_fu_1281_p1 = GELU_TABLE_S_load_4_reg_1687_pp0_iter4_reg;

assign GELU_TABLE_S_load_5_cast_fu_1293_p1 = GELU_TABLE_S_load_5_reg_1697_pp0_iter4_reg;

assign GELU_TABLE_S_load_6_cast_fu_1305_p1 = GELU_TABLE_S_load_6_reg_1707_pp0_iter4_reg;

assign GELU_TABLE_S_load_7_cast_fu_1317_p1 = GELU_TABLE_S_load_7_reg_1717_pp0_iter4_reg;

assign GELU_TABLE_S_load_cast_fu_1229_p1 = GELU_TABLE_S_load_reg_1647_pp0_iter4_reg;

assign GELU_TABLE_address0 = zext_ln131_14_fu_1222_p1;

assign GELU_TABLE_address1 = zext_ln131_12_fu_1218_p1;

assign GELU_TABLE_address2 = zext_ln131_10_fu_1214_p1;

assign GELU_TABLE_address3 = zext_ln131_8_fu_1210_p1;

assign GELU_TABLE_address4 = zext_ln131_6_fu_1206_p1;

assign GELU_TABLE_address5 = zext_ln131_4_fu_1202_p1;

assign GELU_TABLE_address6 = zext_ln131_2_fu_1198_p1;

assign GELU_TABLE_address7 = zext_ln131_fu_1194_p1;

assign GELU_TABLE_load_1_cast_fu_1242_p1 = $signed(GELU_TABLE_load_1_reg_1727);

assign GELU_TABLE_load_2_cast_fu_1254_p1 = $signed(GELU_TABLE_load_2_reg_1732);

assign GELU_TABLE_load_3_cast_fu_1266_p1 = $signed(GELU_TABLE_load_3_reg_1737);

assign GELU_TABLE_load_4_cast_fu_1278_p1 = $signed(GELU_TABLE_load_4_reg_1742);

assign GELU_TABLE_load_5_cast_fu_1290_p1 = $signed(GELU_TABLE_load_5_reg_1747);

assign GELU_TABLE_load_6_cast_fu_1302_p1 = $signed(GELU_TABLE_load_6_reg_1752);

assign GELU_TABLE_load_7_cast_fu_1314_p1 = $signed(GELU_TABLE_load_7_reg_1757);

assign GELU_TABLE_load_cast_fu_1226_p1 = $signed(GELU_TABLE_load_reg_1722);

assign act_stream_din = $signed(tmp_fu_1350_p9);

assign add_ln126_1_fu_454_p2 = ($signed(sext_ln126_1_fu_451_p1) + $signed(29'd26214400));

assign add_ln126_2_fu_487_p2 = ($signed(sext_ln126_2_fu_484_p1) + $signed(29'd26214400));

assign add_ln126_3_fu_520_p2 = ($signed(sext_ln126_3_fu_517_p1) + $signed(29'd26214400));

assign add_ln126_4_fu_553_p2 = ($signed(sext_ln126_4_fu_550_p1) + $signed(29'd26214400));

assign add_ln126_5_fu_586_p2 = ($signed(sext_ln126_5_fu_583_p1) + $signed(29'd26214400));

assign add_ln126_6_fu_619_p2 = ($signed(sext_ln126_6_fu_616_p1) + $signed(29'd26214400));

assign add_ln126_7_fu_652_p2 = ($signed(sext_ln126_7_fu_649_p1) + $signed(29'd26214400));

assign add_ln126_fu_421_p2 = ($signed(sext_ln126_fu_418_p1) + $signed(29'd26214400));

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

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln141_fu_327_p2 == 1'd0) & (mlp_i_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state6_pp0_stage0_iter5 = ((1'b0 == act_stream_full_n) & (icmp_ln141_reg_1382_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_117 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign gelu_val_1_fu_1248_p2 = GELU_TABLE_load_1_cast_fu_1242_p1 << GELU_TABLE_S_load_1_cast_fu_1245_p1;

assign gelu_val_2_fu_1260_p2 = GELU_TABLE_load_2_cast_fu_1254_p1 << GELU_TABLE_S_load_2_cast_fu_1257_p1;

assign gelu_val_3_fu_1272_p2 = GELU_TABLE_load_3_cast_fu_1266_p1 << GELU_TABLE_S_load_3_cast_fu_1269_p1;

assign gelu_val_4_fu_1284_p2 = GELU_TABLE_load_4_cast_fu_1278_p1 << GELU_TABLE_S_load_4_cast_fu_1281_p1;

assign gelu_val_5_fu_1296_p2 = GELU_TABLE_load_5_cast_fu_1290_p1 << GELU_TABLE_S_load_5_cast_fu_1293_p1;

assign gelu_val_6_fu_1308_p2 = GELU_TABLE_load_6_cast_fu_1302_p1 << GELU_TABLE_S_load_6_cast_fu_1305_p1;

assign gelu_val_7_fu_1320_p2 = GELU_TABLE_load_7_cast_fu_1314_p1 << GELU_TABLE_S_load_7_cast_fu_1317_p1;

assign gelu_val_fu_1232_p2 = GELU_TABLE_load_cast_fu_1226_p1 << GELU_TABLE_S_load_cast_fu_1229_p1;

assign icmp_ln141_fu_327_p2 = ((ap_sig_allocacmp_vec_1 == 19'd393216) ? 1'b1 : 1'b0);

assign icmp_ln141_reg_1382_pp0_iter0_reg = icmp_ln141_reg_1382;

assign icmp_ln43_1_fu_478_p2 = (($signed(tmp_4_fu_468_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_2_fu_511_p2 = (($signed(tmp_6_fu_501_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_3_fu_544_p2 = (($signed(tmp_8_fu_534_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_4_fu_577_p2 = (($signed(tmp_10_fu_567_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_5_fu_610_p2 = (($signed(tmp_12_fu_600_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_6_fu_643_p2 = (($signed(tmp_14_fu_633_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_7_fu_676_p2 = (($signed(tmp_16_fu_666_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_fu_445_p2 = (($signed(tmp_2_fu_435_p4) > $signed(3'd0)) ? 1'b1 : 1'b0);

assign lut_idx_1_fu_781_p3 = ((or_ln42_1_fu_777_p2[0:0] == 1'b1) ? select_ln42_4_fu_769_p3 : trunc_ln_fu_746_p4);

assign lut_idx_2_fu_845_p3 = ((or_ln42_2_fu_841_p2[0:0] == 1'b1) ? select_ln42_8_fu_833_p3 : trunc_ln126_5_fu_810_p4);

assign lut_idx_3_fu_909_p3 = ((or_ln42_3_fu_905_p2[0:0] == 1'b1) ? select_ln42_12_fu_897_p3 : trunc_ln126_7_fu_874_p4);

assign lut_idx_4_fu_973_p3 = ((or_ln42_4_fu_969_p2[0:0] == 1'b1) ? select_ln42_16_fu_961_p3 : trunc_ln126_9_fu_938_p4);

assign lut_idx_5_fu_1037_p3 = ((or_ln42_5_fu_1033_p2[0:0] == 1'b1) ? select_ln42_20_fu_1025_p3 : trunc_ln126_s_fu_1002_p4);

assign lut_idx_6_fu_1101_p3 = ((or_ln42_6_fu_1097_p2[0:0] == 1'b1) ? select_ln42_24_fu_1089_p3 : trunc_ln126_1_fu_1066_p4);

assign lut_idx_7_fu_1165_p3 = ((or_ln42_7_fu_1161_p2[0:0] == 1'b1) ? select_ln42_28_fu_1153_p3 : trunc_ln126_3_fu_1130_p4);

assign lut_idx_fu_717_p3 = ((or_ln42_fu_713_p2[0:0] == 1'b1) ? select_ln42_fu_705_p3 : trunc_ln126_2_fu_682_p4);

assign lut_s_idx_1_fu_797_p3 = ((or_ln42_1_fu_777_p2[0:0] == 1'b1) ? select_ln42_6_fu_789_p3 : trunc_ln1_fu_755_p4);

assign lut_s_idx_2_fu_861_p3 = ((or_ln42_2_fu_841_p2[0:0] == 1'b1) ? select_ln42_10_fu_853_p3 : trunc_ln127_5_fu_819_p4);

assign lut_s_idx_3_fu_925_p3 = ((or_ln42_3_fu_905_p2[0:0] == 1'b1) ? select_ln42_14_fu_917_p3 : trunc_ln127_7_fu_883_p4);

assign lut_s_idx_4_fu_989_p3 = ((or_ln42_4_fu_969_p2[0:0] == 1'b1) ? select_ln42_18_fu_981_p3 : trunc_ln127_9_fu_947_p4);

assign lut_s_idx_5_fu_1053_p3 = ((or_ln42_5_fu_1033_p2[0:0] == 1'b1) ? select_ln42_22_fu_1045_p3 : trunc_ln127_s_fu_1011_p4);

assign lut_s_idx_6_fu_1117_p3 = ((or_ln42_6_fu_1097_p2[0:0] == 1'b1) ? select_ln42_26_fu_1109_p3 : trunc_ln127_1_fu_1075_p4);

assign lut_s_idx_7_fu_1181_p3 = ((or_ln42_7_fu_1161_p2[0:0] == 1'b1) ? select_ln42_30_fu_1173_p3 : trunc_ln127_3_fu_1139_p4);

assign lut_s_idx_fu_733_p3 = ((or_ln42_fu_713_p2[0:0] == 1'b1) ? select_ln42_2_fu_725_p3 : trunc_ln127_2_fu_691_p4);

assign or_ln42_1_fu_777_p2 = (tmp_3_reg_1449 | icmp_ln43_1_reg_1455);

assign or_ln42_2_fu_841_p2 = (tmp_5_reg_1466 | icmp_ln43_2_reg_1472);

assign or_ln42_3_fu_905_p2 = (tmp_7_reg_1483 | icmp_ln43_3_reg_1489);

assign or_ln42_4_fu_969_p2 = (tmp_9_reg_1500 | icmp_ln43_4_reg_1506);

assign or_ln42_5_fu_1033_p2 = (tmp_11_reg_1517 | icmp_ln43_5_reg_1523);

assign or_ln42_6_fu_1097_p2 = (tmp_13_reg_1534 | icmp_ln43_6_reg_1540);

assign or_ln42_7_fu_1161_p2 = (tmp_15_reg_1551 | icmp_ln43_7_reg_1557);

assign or_ln42_fu_713_p2 = (tmp_1_reg_1432 | icmp_ln43_reg_1438);

assign select_ln42_10_fu_853_p3 = ((xor_ln42_2_fu_828_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_12_fu_897_p3 = ((xor_ln42_3_fu_892_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_14_fu_917_p3 = ((xor_ln42_3_fu_892_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_16_fu_961_p3 = ((xor_ln42_4_fu_956_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_18_fu_981_p3 = ((xor_ln42_4_fu_956_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_20_fu_1025_p3 = ((xor_ln42_5_fu_1020_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_22_fu_1045_p3 = ((xor_ln42_5_fu_1020_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_24_fu_1089_p3 = ((xor_ln42_6_fu_1084_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_26_fu_1109_p3 = ((xor_ln42_6_fu_1084_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_28_fu_1153_p3 = ((xor_ln42_7_fu_1148_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_2_fu_725_p3 = ((xor_ln42_fu_700_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_30_fu_1173_p3 = ((xor_ln42_7_fu_1148_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_4_fu_769_p3 = ((xor_ln42_1_fu_764_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_6_fu_789_p3 = ((xor_ln42_1_fu_764_p2[0:0] == 1'b1) ? 5'd31 : 5'd0);

assign select_ln42_8_fu_833_p3 = ((xor_ln42_2_fu_828_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign select_ln42_fu_705_p3 = ((xor_ln42_fu_700_p2[0:0] == 1'b1) ? 12'd4095 : 12'd0);

assign sext_ln126_1_fu_451_p1 = $signed(x_5_reg_1391);

assign sext_ln126_2_fu_484_p1 = $signed(x_7_reg_1396);

assign sext_ln126_3_fu_517_p1 = $signed(x_9_reg_1401);

assign sext_ln126_4_fu_550_p1 = $signed(x_11_reg_1406);

assign sext_ln126_5_fu_583_p1 = $signed(x_12_reg_1411);

assign sext_ln126_6_fu_616_p1 = $signed(x_13_reg_1416);

assign sext_ln126_7_fu_649_p1 = $signed(x_14_reg_1421);

assign sext_ln126_fu_418_p1 = $signed(value_reg_1386);

assign sext_ln132_fu_1238_p1 = $signed(gelu_val_fu_1232_p2);

assign sext_ln150_1_fu_1330_p1 = $signed(gelu_val_2_fu_1260_p2);

assign sext_ln150_2_fu_1334_p1 = $signed(gelu_val_3_fu_1272_p2);

assign sext_ln150_3_fu_1338_p1 = $signed(gelu_val_4_fu_1284_p2);

assign sext_ln150_4_fu_1342_p1 = $signed(gelu_val_5_fu_1296_p2);

assign sext_ln150_5_fu_1346_p1 = $signed(gelu_val_6_fu_1308_p2);

assign sext_ln150_fu_1326_p1 = $signed(gelu_val_1_fu_1248_p2);

assign tmp_10_fu_567_p4 = {{add_ln126_4_fu_553_p2[28:26]}};

assign tmp_12_fu_600_p4 = {{add_ln126_5_fu_586_p2[28:26]}};

assign tmp_14_fu_633_p4 = {{add_ln126_6_fu_619_p2[28:26]}};

assign tmp_16_fu_666_p4 = {{add_ln126_7_fu_652_p2[28:26]}};

assign tmp_2_fu_435_p4 = {{add_ln126_fu_421_p2[28:26]}};

assign tmp_4_fu_468_p4 = {{add_ln126_1_fu_454_p2[28:26]}};

assign tmp_6_fu_501_p4 = {{add_ln126_2_fu_487_p2[28:26]}};

assign tmp_8_fu_534_p4 = {{add_ln126_3_fu_520_p2[28:26]}};

assign tmp_fu_1350_p9 = {{{{{{{{gelu_val_7_fu_1320_p2}, {sext_ln150_5_fu_1346_p1}}, {sext_ln150_4_fu_1342_p1}}, {sext_ln150_3_fu_1338_p1}}, {sext_ln150_2_fu_1334_p1}}, {sext_ln150_1_fu_1330_p1}}, {sext_ln150_fu_1326_p1}}, {sext_ln132_fu_1238_p1}};

assign trunc_ln126_1_fu_1066_p4 = {{add_ln126_6_reg_1528[25:14]}};

assign trunc_ln126_2_fu_682_p4 = {{add_ln126_reg_1426[25:14]}};

assign trunc_ln126_3_fu_1130_p4 = {{add_ln126_7_reg_1545[25:14]}};

assign trunc_ln126_5_fu_810_p4 = {{add_ln126_2_reg_1460[25:14]}};

assign trunc_ln126_7_fu_874_p4 = {{add_ln126_3_reg_1477[25:14]}};

assign trunc_ln126_9_fu_938_p4 = {{add_ln126_4_reg_1494[25:14]}};

assign trunc_ln126_s_fu_1002_p4 = {{add_ln126_5_reg_1511[25:14]}};

assign trunc_ln127_1_fu_1075_p4 = {{add_ln126_6_reg_1528[25:21]}};

assign trunc_ln127_2_fu_691_p4 = {{add_ln126_reg_1426[25:21]}};

assign trunc_ln127_3_fu_1139_p4 = {{add_ln126_7_reg_1545[25:21]}};

assign trunc_ln127_5_fu_819_p4 = {{add_ln126_2_reg_1460[25:21]}};

assign trunc_ln127_7_fu_883_p4 = {{add_ln126_3_reg_1477[25:21]}};

assign trunc_ln127_9_fu_947_p4 = {{add_ln126_4_reg_1494[25:21]}};

assign trunc_ln127_s_fu_1011_p4 = {{add_ln126_5_reg_1511[25:21]}};

assign trunc_ln1_fu_755_p4 = {{add_ln126_1_reg_1443[25:21]}};

assign trunc_ln_fu_746_p4 = {{add_ln126_1_reg_1443[25:14]}};

assign value_fu_339_p1 = mlp_i_stream_TDATA[27:0];

assign vec_2_fu_333_p2 = (ap_sig_allocacmp_vec_1 + 19'd1);

assign xor_ln42_1_fu_764_p2 = (tmp_3_reg_1449 ^ 1'd1);

assign xor_ln42_2_fu_828_p2 = (tmp_5_reg_1466 ^ 1'd1);

assign xor_ln42_3_fu_892_p2 = (tmp_7_reg_1483 ^ 1'd1);

assign xor_ln42_4_fu_956_p2 = (tmp_9_reg_1500 ^ 1'd1);

assign xor_ln42_5_fu_1020_p2 = (tmp_11_reg_1517 ^ 1'd1);

assign xor_ln42_6_fu_1084_p2 = (tmp_13_reg_1534 ^ 1'd1);

assign xor_ln42_7_fu_1148_p2 = (tmp_15_reg_1551 ^ 1'd1);

assign xor_ln42_fu_700_p2 = (tmp_1_reg_1432 ^ 1'd1);

assign zext_ln131_10_fu_1214_p1 = lut_idx_5_reg_1612;

assign zext_ln131_11_fu_1061_p1 = lut_s_idx_5_fu_1053_p3;

assign zext_ln131_12_fu_1218_p1 = lut_idx_6_reg_1622;

assign zext_ln131_13_fu_1125_p1 = lut_s_idx_6_fu_1117_p3;

assign zext_ln131_14_fu_1222_p1 = lut_idx_7_reg_1632;

assign zext_ln131_15_fu_1189_p1 = lut_s_idx_7_fu_1181_p3;

assign zext_ln131_1_fu_741_p1 = lut_s_idx_fu_733_p3;

assign zext_ln131_2_fu_1198_p1 = lut_idx_1_reg_1572;

assign zext_ln131_3_fu_805_p1 = lut_s_idx_1_fu_797_p3;

assign zext_ln131_4_fu_1202_p1 = lut_idx_2_reg_1582;

assign zext_ln131_5_fu_869_p1 = lut_s_idx_2_fu_861_p3;

assign zext_ln131_6_fu_1206_p1 = lut_idx_3_reg_1592;

assign zext_ln131_7_fu_933_p1 = lut_s_idx_3_fu_925_p3;

assign zext_ln131_8_fu_1210_p1 = lut_idx_4_reg_1602;

assign zext_ln131_9_fu_997_p1 = lut_s_idx_4_fu_989_p3;

assign zext_ln131_fu_1194_p1 = lut_idx_reg_1562;

endmodule //SILU_GELU_vit_gelu_to_common
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_do_adapt_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        silu_stream_dout,
        silu_stream_num_data_valid,
        silu_stream_fifo_cap,
        silu_stream_empty_n,
        silu_stream_read,
        silu_stream1_din,
        silu_stream1_num_data_valid,
        silu_stream1_fifo_cap,
        silu_stream1_full_n,
        silu_stream1_write
);

parameter    ap_ST_fsm_state1 = 2'd1;
parameter    ap_ST_fsm_state2 = 2'd2;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [20:0] silu_stream_dout;
input  [2:0] silu_stream_num_data_valid;
input  [2:0] silu_stream_fifo_cap;
input   silu_stream_empty_n;
output   silu_stream_read;
output  [167:0] silu_stream1_din;
input  [4:0] silu_stream1_num_data_valid;
input  [4:0] silu_stream1_fifo_cap;
input   silu_stream1_full_n;
output   silu_stream1_write;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg silu_stream_read;
reg silu_stream1_write;

reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [1:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire    grp_pack_fu_16_ap_start;
wire    grp_pack_fu_16_ap_done;
wire    grp_pack_fu_16_ap_idle;
wire    grp_pack_fu_16_ap_ready;
wire    grp_pack_fu_16_silu_stream_read;
wire   [167:0] grp_pack_fu_16_silu_stream1_din;
wire    grp_pack_fu_16_silu_stream1_write;
reg    grp_pack_fu_16_ap_start_reg;
reg    ap_block_state1_ignore_call2;
wire    ap_CS_fsm_state2;
reg   [1:0] ap_NS_fsm;
reg    ap_block_state1;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 2'd1;
//#0 grp_pack_fu_16_ap_start_reg = 1'b0;
end

SILU_GELU_pack grp_pack_fu_16(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_pack_fu_16_ap_start),
    .ap_done(grp_pack_fu_16_ap_done),
    .ap_idle(grp_pack_fu_16_ap_idle),
    .ap_ready(grp_pack_fu_16_ap_ready),
    .silu_stream_dout(silu_stream_dout),
    .silu_stream_num_data_valid(3'd0),
    .silu_stream_fifo_cap(3'd0),
    .silu_stream_empty_n(silu_stream_empty_n),
    .silu_stream_read(grp_pack_fu_16_silu_stream_read),
    .silu_stream1_din(grp_pack_fu_16_silu_stream1_din),
    .silu_stream1_num_data_valid(5'd0),
    .silu_stream1_fifo_cap(5'd0),
    .silu_stream1_full_n(silu_stream1_full_n),
    .silu_stream1_write(grp_pack_fu_16_silu_stream1_write)
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
        end else if (((grp_pack_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_pack_fu_16_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state1_ignore_call2) & (1'b1 == ap_CS_fsm_state1))) begin
            grp_pack_fu_16_ap_start_reg <= 1'b1;
        end else if ((grp_pack_fu_16_ap_ready == 1'b1)) begin
            grp_pack_fu_16_ap_start_reg <= 1'b0;
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

always @ (*) begin
    if ((grp_pack_fu_16_ap_done == 1'b0)) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if (((grp_pack_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
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
    if (((grp_pack_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        silu_stream1_write = grp_pack_fu_16_silu_stream1_write;
    end else begin
        silu_stream1_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        silu_stream_read = grp_pack_fu_16_silu_stream_read;
    end else begin
        silu_stream_read = 1'b0;
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
            if (((grp_pack_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

always @ (*) begin
    ap_block_state1 = ((ap_start == 1'b0) | (ap_done_reg == 1'b1));
end

always @ (*) begin
    ap_block_state1_ignore_call2 = ((ap_start == 1'b0) | (ap_done_reg == 1'b1));
end

assign grp_pack_fu_16_ap_start = grp_pack_fu_16_ap_start_reg;

assign silu_stream1_din = grp_pack_fu_16_silu_stream1_din;

endmodule //SILU_GELU_do_adapt_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module SILU_GELU_fifo_w21_d2_S
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 21,
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
    SILU_GELU_fifo_w21_d2_S_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_fifo_w21_d2_S_ShiftReg (
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


module SILU_GELU_fifo_w21_d2_S_ShiftReg
#(parameter
    DATA_WIDTH  = 21,
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

module SILU_GELU_unpk (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        g_stream_dout,
        g_stream_num_data_valid,
        g_stream_fifo_cap,
        g_stream_empty_n,
        g_stream_read,
        adpt_stream_din,
        adpt_stream_num_data_valid,
        adpt_stream_fifo_cap,
        adpt_stream_full_n,
        adpt_stream_write
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [167:0] g_stream_dout;
input  [4:0] g_stream_num_data_valid;
input  [4:0] g_stream_fifo_cap;
input   g_stream_empty_n;
output   g_stream_read;
output  [20:0] adpt_stream_din;
input  [2:0] adpt_stream_num_data_valid;
input  [2:0] adpt_stream_fifo_cap;
input   adpt_stream_full_n;
output   adpt_stream_write;

reg ap_idle;
reg g_stream_read;
reg adpt_stream_write;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
reg   [0:0] icmp_ln50_reg_582;
reg   [0:0] icmp_ln50_reg_582_pp0_iter1_reg;
reg   [0:0] icmp_ln55_reg_592;
wire   [0:0] icmp_ln55_reg_592_pp0_iter1_reg;
reg    ap_predicate_op60_read_state3;
reg    ap_block_state3_pp0_stage0_iter2;
wire    ap_CS_iter2_fsm_state3;
wire   [0:0] icmp_ln50_fu_275_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    g_stream_blk_n;
reg    adpt_stream_blk_n;
wire   [0:0] icmp_ln50_reg_582_pp0_iter0_reg;
wire   [0:0] icmp_ln51_fu_287_p2;
reg   [0:0] icmp_ln51_reg_586;
wire   [0:0] icmp_ln55_fu_350_p2;
reg   [20:0] ap_phi_mux_p_0_0_7_0_0_083_phi_fu_148_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_7_0_0_083_reg_145;
reg   [20:0] ap_phi_mux_p_0_0_6_0_0_081_phi_fu_158_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_6_0_0_081_reg_155;
reg   [20:0] ap_phi_mux_p_0_0_5_0_0_079_phi_fu_167_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_5_0_0_079_reg_164;
reg   [20:0] ap_phi_mux_p_0_0_4_0_0_077_phi_fu_176_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_4_0_0_077_reg_173;
reg   [20:0] ap_phi_mux_p_0_0_3_0_0_075_phi_fu_185_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_3_0_0_075_reg_182;
reg   [20:0] ap_phi_mux_p_0_0_2_0_0_073_phi_fu_194_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_2_0_0_073_reg_191;
reg   [20:0] ap_phi_mux_p_0_0_1_0_0_071_phi_fu_203_p4;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_1_0_0_071_reg_200;
reg   [20:0] ap_phi_mux_p_0_0_0_0_0_069_phi_fu_212_p4;
wire   [20:0] trunc_ln55_fu_395_p1;
wire   [20:0] ap_phi_reg_pp0_iter2_p_0_0_0_0_0_069_reg_209;
reg   [3:0] t_fu_92;
wire   [3:0] t_1_fu_356_p2;
wire    ap_loop_init;
reg   [12:0] indvar_flatten_fu_96;
wire   [12:0] select_ln51_1_fu_299_p3;
reg   [12:0] ap_sig_allocacmp_indvar_flatten_load;
reg   [14:0] indvar_flatten24_fu_100;
wire   [14:0] add_ln50_fu_281_p2;
reg   [14:0] ap_sig_allocacmp_indvar_flatten24_load;
reg   [20:0] p_0_0_0_0_0_068_fu_104;
reg   [20:0] p_0_0_1_0_0_070_fu_108;
reg   [20:0] p_0_0_2_0_0_072_fu_112;
reg   [20:0] p_0_0_3_0_0_074_fu_116;
reg   [20:0] p_0_0_4_0_0_076_fu_120;
reg   [20:0] p_0_0_5_0_0_078_fu_124;
reg   [20:0] p_0_0_6_0_0_080_fu_128;
wire   [12:0] add_ln51_fu_293_p2;
wire   [0:0] icmp_ln52_fu_325_p2;
wire   [0:0] xor_ln50_fu_320_p2;
wire   [0:0] and_ln50_fu_331_p2;
wire   [0:0] or_ln51_fu_337_p2;
wire   [3:0] select_ln51_fu_342_p3;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
reg    ap_ST_iter2_fsm_state3_blk;
wire    ap_start_int;
reg    ap_condition_85;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 t_fu_92 = 4'd0;
//#0 indvar_flatten_fu_96 = 13'd0;
//#0 indvar_flatten24_fu_100 = 15'd0;
//#0 p_0_0_0_0_0_068_fu_104 = 21'd0;
//#0 p_0_0_1_0_0_070_fu_108 = 21'd0;
//#0 p_0_0_2_0_0_072_fu_112 = 21'd0;
//#0 p_0_0_3_0_0_074_fu_116 = 21'd0;
//#0 p_0_0_4_0_0_076_fu_120 = 21'd0;
//#0 p_0_0_5_0_0_078_fu_124 = 21'd0;
//#0 p_0_0_6_0_0_080_fu_128 = 21'd0;
//#0 ap_done_reg = 1'b0;
end

SILU_GELU_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_85)) begin
        if ((icmp_ln50_fu_275_p2 == 1'd0)) begin
            indvar_flatten24_fu_100 <= add_ln50_fu_281_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten24_fu_100 <= 15'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_85)) begin
        if ((icmp_ln50_fu_275_p2 == 1'd0)) begin
            indvar_flatten_fu_96 <= select_ln51_1_fu_299_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_96 <= 13'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_0_0_0_068_fu_104 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_0_0_0_068_fu_104 <= ap_phi_mux_p_0_0_1_0_0_071_phi_fu_203_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_1_0_0_070_fu_108 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_1_0_0_070_fu_108 <= ap_phi_mux_p_0_0_2_0_0_073_phi_fu_194_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_2_0_0_072_fu_112 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_2_0_0_072_fu_112 <= ap_phi_mux_p_0_0_3_0_0_075_phi_fu_185_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_3_0_0_074_fu_116 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_3_0_0_074_fu_116 <= ap_phi_mux_p_0_0_4_0_0_077_phi_fu_176_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_4_0_0_076_fu_120 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_4_0_0_076_fu_120 <= ap_phi_mux_p_0_0_5_0_0_079_phi_fu_167_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_5_0_0_078_fu_124 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_5_0_0_078_fu_124 <= ap_phi_mux_p_0_0_6_0_0_081_phi_fu_158_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        p_0_0_6_0_0_080_fu_128 <= 21'd0;
    end else if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        p_0_0_6_0_0_080_fu_128 <= ap_phi_mux_p_0_0_7_0_0_083_phi_fu_148_p4;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        t_fu_92 <= 4'd0;
    end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln50_reg_582_pp0_iter0_reg == 1'd0))) begin
        t_fu_92 <= t_1_fu_356_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln50_reg_582 <= icmp_ln50_fu_275_p2;
        icmp_ln51_reg_586 <= icmp_ln51_fu_287_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        icmp_ln50_reg_582_pp0_iter1_reg <= icmp_ln50_reg_582;
        icmp_ln55_reg_592 <= icmp_ln55_fu_350_p2;
    end
end

always @ (*) begin
    if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        adpt_stream_blk_n = adpt_stream_full_n;
    end else begin
        adpt_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        adpt_stream_write = 1'b1;
    end else begin
        adpt_stream_write = 1'b0;
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
    if ((1'b1 == ap_block_state3_pp0_stage0_iter2)) begin
        ap_ST_iter2_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (icmp_ln50_fu_275_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_0_0_0_069_phi_fu_212_p4 = p_0_0_0_0_0_068_fu_104;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_0_0_0_069_phi_fu_212_p4 = trunc_ln55_fu_395_p1;
        end else begin
            ap_phi_mux_p_0_0_0_0_0_069_phi_fu_212_p4 = ap_phi_reg_pp0_iter2_p_0_0_0_0_0_069_reg_209;
        end
    end else begin
        ap_phi_mux_p_0_0_0_0_0_069_phi_fu_212_p4 = ap_phi_reg_pp0_iter2_p_0_0_0_0_0_069_reg_209;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_1_0_0_071_phi_fu_203_p4 = p_0_0_1_0_0_070_fu_108;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_1_0_0_071_phi_fu_203_p4 = {{g_stream_dout[41:21]}};
        end else begin
            ap_phi_mux_p_0_0_1_0_0_071_phi_fu_203_p4 = ap_phi_reg_pp0_iter2_p_0_0_1_0_0_071_reg_200;
        end
    end else begin
        ap_phi_mux_p_0_0_1_0_0_071_phi_fu_203_p4 = ap_phi_reg_pp0_iter2_p_0_0_1_0_0_071_reg_200;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_2_0_0_073_phi_fu_194_p4 = p_0_0_2_0_0_072_fu_112;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_2_0_0_073_phi_fu_194_p4 = {{g_stream_dout[62:42]}};
        end else begin
            ap_phi_mux_p_0_0_2_0_0_073_phi_fu_194_p4 = ap_phi_reg_pp0_iter2_p_0_0_2_0_0_073_reg_191;
        end
    end else begin
        ap_phi_mux_p_0_0_2_0_0_073_phi_fu_194_p4 = ap_phi_reg_pp0_iter2_p_0_0_2_0_0_073_reg_191;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_3_0_0_075_phi_fu_185_p4 = p_0_0_3_0_0_074_fu_116;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_3_0_0_075_phi_fu_185_p4 = {{g_stream_dout[83:63]}};
        end else begin
            ap_phi_mux_p_0_0_3_0_0_075_phi_fu_185_p4 = ap_phi_reg_pp0_iter2_p_0_0_3_0_0_075_reg_182;
        end
    end else begin
        ap_phi_mux_p_0_0_3_0_0_075_phi_fu_185_p4 = ap_phi_reg_pp0_iter2_p_0_0_3_0_0_075_reg_182;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_4_0_0_077_phi_fu_176_p4 = p_0_0_4_0_0_076_fu_120;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_4_0_0_077_phi_fu_176_p4 = {{g_stream_dout[104:84]}};
        end else begin
            ap_phi_mux_p_0_0_4_0_0_077_phi_fu_176_p4 = ap_phi_reg_pp0_iter2_p_0_0_4_0_0_077_reg_173;
        end
    end else begin
        ap_phi_mux_p_0_0_4_0_0_077_phi_fu_176_p4 = ap_phi_reg_pp0_iter2_p_0_0_4_0_0_077_reg_173;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_5_0_0_079_phi_fu_167_p4 = p_0_0_5_0_0_078_fu_124;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_5_0_0_079_phi_fu_167_p4 = {{g_stream_dout[125:105]}};
        end else begin
            ap_phi_mux_p_0_0_5_0_0_079_phi_fu_167_p4 = ap_phi_reg_pp0_iter2_p_0_0_5_0_0_079_reg_164;
        end
    end else begin
        ap_phi_mux_p_0_0_5_0_0_079_phi_fu_167_p4 = ap_phi_reg_pp0_iter2_p_0_0_5_0_0_079_reg_164;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_6_0_0_081_phi_fu_158_p4 = p_0_0_6_0_0_080_fu_128;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_6_0_0_081_phi_fu_158_p4 = {{g_stream_dout[146:126]}};
        end else begin
            ap_phi_mux_p_0_0_6_0_0_081_phi_fu_158_p4 = ap_phi_reg_pp0_iter2_p_0_0_6_0_0_081_reg_155;
        end
    end else begin
        ap_phi_mux_p_0_0_6_0_0_081_phi_fu_158_p4 = ap_phi_reg_pp0_iter2_p_0_0_6_0_0_081_reg_155;
    end
end

always @ (*) begin
    if ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0)) begin
        if ((icmp_ln55_reg_592_pp0_iter1_reg == 1'd0)) begin
            ap_phi_mux_p_0_0_7_0_0_083_phi_fu_148_p4 = 21'd0;
        end else if ((icmp_ln55_reg_592 == 1'd1)) begin
            ap_phi_mux_p_0_0_7_0_0_083_phi_fu_148_p4 = {{g_stream_dout[167:147]}};
        end else begin
            ap_phi_mux_p_0_0_7_0_0_083_phi_fu_148_p4 = ap_phi_reg_pp0_iter2_p_0_0_7_0_0_083_reg_145;
        end
    end else begin
        ap_phi_mux_p_0_0_7_0_0_083_phi_fu_148_p4 = ap_phi_reg_pp0_iter2_p_0_0_7_0_0_083_reg_145;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten24_load = 15'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten24_load = indvar_flatten24_fu_100;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 13'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_96;
    end
end

always @ (*) begin
    if (((ap_predicate_op60_read_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        g_stream_blk_n = g_stream_empty_n;
    end else begin
        g_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_predicate_op60_read_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        g_stream_read = 1'b1;
    end else begin
        g_stream_read = 1'b0;
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
            if ((~((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2)) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if (((1'b0 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state3_pp0_stage0_iter2)) | ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd1) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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

assign add_ln50_fu_281_p2 = (ap_sig_allocacmp_indvar_flatten24_load + 15'd1);

assign add_ln51_fu_293_p2 = (ap_sig_allocacmp_indvar_flatten_load + 13'd1);

assign adpt_stream_din = ap_phi_mux_p_0_0_0_0_0_069_phi_fu_212_p4;

assign and_ln50_fu_331_p2 = (xor_ln50_fu_320_p2 & icmp_ln52_fu_325_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state3_pp0_stage0_iter2 = (((ap_predicate_op60_read_state3 == 1'b1) & (g_stream_empty_n == 1'b0)) | ((icmp_ln50_reg_582_pp0_iter1_reg == 1'd0) & (1'b0 == adpt_stream_full_n)));
end

always @ (*) begin
    ap_condition_85 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_phi_reg_pp0_iter2_p_0_0_0_0_0_069_reg_209 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_1_0_0_071_reg_200 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_2_0_0_073_reg_191 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_3_0_0_075_reg_182 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_4_0_0_077_reg_173 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_5_0_0_079_reg_164 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_6_0_0_081_reg_155 = 'bx;

assign ap_phi_reg_pp0_iter2_p_0_0_7_0_0_083_reg_145 = 'bx;

always @ (*) begin
    ap_predicate_op60_read_state3 = ((icmp_ln55_reg_592 == 1'd1) & (icmp_ln50_reg_582_pp0_iter1_reg == 1'd0));
end

assign icmp_ln50_fu_275_p2 = ((ap_sig_allocacmp_indvar_flatten24_load == 15'd20480) ? 1'b1 : 1'b0);

assign icmp_ln50_reg_582_pp0_iter0_reg = icmp_ln50_reg_582;

assign icmp_ln51_fu_287_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 13'd2560) ? 1'b1 : 1'b0);

assign icmp_ln52_fu_325_p2 = ((t_fu_92 == 4'd8) ? 1'b1 : 1'b0);

assign icmp_ln55_fu_350_p2 = ((select_ln51_fu_342_p3 == 4'd0) ? 1'b1 : 1'b0);

assign icmp_ln55_reg_592_pp0_iter1_reg = icmp_ln55_reg_592;

assign or_ln51_fu_337_p2 = (icmp_ln51_reg_586 | and_ln50_fu_331_p2);

assign select_ln51_1_fu_299_p3 = ((icmp_ln51_fu_287_p2[0:0] == 1'b1) ? 13'd1 : add_ln51_fu_293_p2);

assign select_ln51_fu_342_p3 = ((or_ln51_fu_337_p2[0:0] == 1'b1) ? 4'd0 : t_fu_92);

assign t_1_fu_356_p2 = (select_ln51_fu_342_p3 + 4'd1);

assign trunc_ln55_fu_395_p1 = g_stream_dout[20:0];

assign xor_ln50_fu_320_p2 = (icmp_ln51_reg_586 ^ 1'd1);

endmodule //SILU_GELU_unpk
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_shared_mlp_quantize4 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        mode_dout,
        mode_num_data_valid,
        mode_fifo_cap,
        mode_empty_n,
        mode_read,
        act_stream_i_dout,
        act_stream_i_num_data_valid,
        act_stream_i_fifo_cap,
        act_stream_i_empty_n,
        act_stream_i_read,
        q_stream_TDATA,
        q_stream_TVALID,
        q_stream_TREADY,
        s_stream_TDATA,
        s_stream_TVALID,
        s_stream_TREADY
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
input  [0:0] mode_dout;
input  [2:0] mode_num_data_valid;
input  [2:0] mode_fifo_cap;
input   mode_empty_n;
output   mode_read;
input  [215:0] act_stream_i_dout;
input  [5:0] act_stream_i_num_data_valid;
input  [5:0] act_stream_i_fifo_cap;
input   act_stream_i_empty_n;
output   act_stream_i_read;
output  [63:0] q_stream_TDATA;
output   q_stream_TVALID;
input   q_stream_TREADY;
output  [7:0] s_stream_TDATA;
output   s_stream_TVALID;
input   s_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg mode_read;
reg act_stream_i_read;

reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    mode_blk_n;
reg   [0:0] mode_read_reg_67;
reg    ap_block_state1;
wire   [17:0] select_ln175_fu_59_p3;
reg   [17:0] select_ln175_reg_72;
wire    ap_CS_fsm_state2;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_idle;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_ready;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_act_stream_i_read;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TREADY;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TREADY;
wire   [63:0] grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TDATA;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TVALID;
wire   [7:0] grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TDATA;
wire    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TVALID;
reg    grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start_reg;
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
//#0 grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start_reg = 1'b0;
end

SILU_GELU_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1 grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start),
    .ap_done(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done),
    .ap_idle(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_idle),
    .ap_ready(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_ready),
    .act_stream_i_dout(act_stream_i_dout),
    .act_stream_i_num_data_valid(6'd0),
    .act_stream_i_fifo_cap(6'd0),
    .act_stream_i_empty_n(act_stream_i_empty_n),
    .act_stream_i_read(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_act_stream_i_read),
    .q_stream_TREADY(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TREADY),
    .s_stream_TREADY(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TREADY),
    .select_ln175(select_ln175_reg_72),
    .q_stream_TDATA(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TDATA),
    .q_stream_TVALID(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TVALID),
    .s_stream_TDATA(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TDATA),
    .s_stream_TVALID(grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TVALID)
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
        end else if (((grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start_reg <= 1'b1;
        end else if ((grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_ready == 1'b1)) begin
            grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        mode_read_reg_67 <= mode_dout;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        select_ln175_reg_72[9] <= select_ln175_fu_59_p3[9];
select_ln175_reg_72[11] <= select_ln175_fu_59_p3[11];
select_ln175_reg_72[17] <= select_ln175_fu_59_p3[17];
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        act_stream_i_read = grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_act_stream_i_read;
    end else begin
        act_stream_i_read = 1'b0;
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
    if ((grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
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
    if (((grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        mode_blk_n = mode_empty_n;
    end else begin
        mode_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        mode_read = 1'b1;
    end else begin
        mode_read = 1'b0;
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
            if (((grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
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
    ap_block_state1 = ((ap_start == 1'b0) | (mode_empty_n == 1'b0) | (ap_done_reg == 1'b1));
end

assign grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start = grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_ap_start_reg;

assign grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TREADY = (q_stream_TREADY & ap_CS_fsm_state3);

assign grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TREADY = (s_stream_TREADY & ap_CS_fsm_state3);

assign q_stream_TDATA = grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TDATA;

assign q_stream_TVALID = grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_q_stream_TVALID;

assign s_stream_TDATA = grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TDATA;

assign s_stream_TVALID = grp_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1_fu_48_s_stream_TVALID;

assign select_ln175_fu_59_p3 = ((mode_read_reg_67[0:0] == 1'b1) ? 18'd131072 : 18'd2560);

always @ (posedge ap_clk) begin
    select_ln175_reg_72[8:0] <= 9'b000000000;
    select_ln175_reg_72[10:10] <= 1'b0;
    select_ln175_reg_72[16:12] <= 5'b00000;
end

endmodule //SILU_GELU_shared_mlp_quantize4
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_llm_split_ug (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        mlp_i_stream_TVALID,
        g_stream_din,
        g_stream_num_data_valid,
        g_stream_fifo_cap,
        g_stream_full_n,
        g_stream_write,
        u_stream_din,
        u_stream_num_data_valid,
        u_stream_fifo_cap,
        u_stream_full_n,
        u_stream_write,
        start_out,
        start_write,
        mlp_i_stream_TDATA,
        mlp_i_stream_TREADY
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input   mlp_i_stream_TVALID;
output  [167:0] g_stream_din;
input  [4:0] g_stream_num_data_valid;
input  [4:0] g_stream_fifo_cap;
input   g_stream_full_n;
output   g_stream_write;
output  [167:0] u_stream_din;
input  [4:0] u_stream_num_data_valid;
input  [4:0] u_stream_fifo_cap;
input   u_stream_full_n;
output   u_stream_write;
output   start_out;
output   start_write;
input  [223:0] mlp_i_stream_TDATA;
output   mlp_i_stream_TREADY;

reg ap_idle;
reg g_stream_write;
reg u_stream_write;
reg start_write;
reg mlp_i_stream_TREADY;

reg    real_start;
reg    start_once_reg;
reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire    internal_ap_ready;
wire   [0:0] icmp_ln58_fu_154_p2;
reg    ap_done_reg;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
reg   [0:0] icmp_ln58_reg_389;
reg   [0:0] icmp_ln58_reg_389_pp0_iter1_reg;
reg   [0:0] cmp15_reg_440;
reg    ap_predicate_op57_write_state3;
reg    ap_predicate_op59_write_state3;
reg    ap_block_state3_pp0_stage0_iter2;
wire    ap_CS_iter2_fsm_state3;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    mlp_i_stream_TDATA_blk_n;
reg    u_stream_blk_n;
reg    g_stream_blk_n;
wire   [0:0] icmp_ln58_reg_389_pp0_iter0_reg;
wire   [0:0] icmp_ln59_fu_166_p2;
reg   [0:0] icmp_ln59_reg_393;
wire   [20:0] out_fu_172_p1;
reg   [20:0] out_reg_400;
reg   [20:0] tmp_reg_405;
reg   [20:0] tmp_2_reg_410;
reg   [20:0] tmp_3_reg_415;
reg   [20:0] tmp_4_reg_420;
reg   [20:0] tmp_5_reg_425;
reg   [20:0] tmp_6_reg_430;
reg   [20:0] tmp_7_reg_435;
wire   [0:0] cmp15_fu_314_p2;
wire   [167:0] or_ln71_s_fu_320_p9;
reg   [167:0] or_ln71_s_reg_444;
reg   [3:0] t_fu_92;
wire   [3:0] t_5_fu_343_p3;
wire    ap_loop_init;
reg    ap_loop_init_pp0_iter1_reg;
reg   [3:0] ap_sig_allocacmp_t_4;
reg   [1:0] ug_fu_96;
wire   [1:0] select_ln59_fu_306_p3;
reg   [1:0] ap_sig_allocacmp_ug_load;
reg   [5:0] indvar_flatten_fu_100;
wire   [5:0] select_ln59_1_fu_252_p3;
reg   [5:0] ap_sig_allocacmp_indvar_flatten_load;
reg   [12:0] indvar_flatten11_fu_104;
wire   [12:0] add_ln58_fu_160_p2;
reg   [12:0] ap_sig_allocacmp_indvar_flatten11_load;
wire   [5:0] add_ln59_1_fu_246_p2;
wire   [0:0] icmp_ln60_fu_288_p2;
wire   [0:0] xor_ln58_fu_283_p2;
wire   [1:0] select_ln58_fu_276_p3;
wire   [0:0] and_ln58_fu_294_p2;
wire   [1:0] ug_2_fu_300_p2;
wire   [0:0] or_ln60_fu_338_p2;
wire   [3:0] add_ln60_fu_332_p2;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
reg    ap_ST_iter2_fsm_state3_blk;
wire    ap_start_int;
reg    ap_condition_107;
reg    ap_condition_141;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 start_once_reg = 1'b0;
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_done_reg = 1'b0;
//#0 t_fu_92 = 4'd0;
//#0 ug_fu_96 = 2'd0;
//#0 indvar_flatten_fu_100 = 6'd0;
//#0 indvar_flatten11_fu_104 = 13'd0;
end

SILU_GELU_flow_control_loop_pipe flow_control_loop_pipe_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(real_start),
    .ap_ready(internal_ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int),
    .ap_continue(ap_continue)
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
        end else if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((internal_ap_ready == 1'b0) & (real_start == 1'b1))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_107)) begin
        if ((icmp_ln58_fu_154_p2 == 1'd0)) begin
            indvar_flatten11_fu_104 <= add_ln58_fu_160_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten11_fu_104 <= 13'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_107)) begin
        if ((icmp_ln58_fu_154_p2 == 1'd0)) begin
            indvar_flatten_fu_100 <= select_ln59_1_fu_252_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_100 <= 6'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_141)) begin
        if ((icmp_ln58_reg_389_pp0_iter0_reg == 1'd0)) begin
            t_fu_92 <= t_5_fu_343_p3;
        end else if ((ap_loop_init_pp0_iter1_reg == 1'b1)) begin
            t_fu_92 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_141)) begin
        if ((icmp_ln58_reg_389_pp0_iter0_reg == 1'd0)) begin
            ug_fu_96 <= select_ln59_fu_306_p3;
        end else if ((ap_loop_init_pp0_iter1_reg == 1'b1)) begin
            ug_fu_96 <= 2'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        ap_loop_init_pp0_iter1_reg <= ap_loop_init;
        icmp_ln58_reg_389 <= icmp_ln58_fu_154_p2;
        icmp_ln59_reg_393 <= icmp_ln59_fu_166_p2;
        out_reg_400 <= out_fu_172_p1;
        tmp_2_reg_410 <= {{mlp_i_stream_TDATA[76:56]}};
        tmp_3_reg_415 <= {{mlp_i_stream_TDATA[104:84]}};
        tmp_4_reg_420 <= {{mlp_i_stream_TDATA[132:112]}};
        tmp_5_reg_425 <= {{mlp_i_stream_TDATA[160:140]}};
        tmp_6_reg_430 <= {{mlp_i_stream_TDATA[188:168]}};
        tmp_7_reg_435 <= {{mlp_i_stream_TDATA[216:196]}};
        tmp_reg_405 <= {{mlp_i_stream_TDATA[48:28]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        cmp15_reg_440 <= cmp15_fu_314_p2;
        icmp_ln58_reg_389_pp0_iter1_reg <= icmp_ln58_reg_389;
        or_ln71_s_reg_444 <= or_ln71_s_fu_320_p9;
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
    if ((1'b1 == ap_block_state3_pp0_stage0_iter2)) begin
        ap_ST_iter2_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (icmp_ln58_fu_154_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten11_load = 13'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten11_load = indvar_flatten11_fu_104;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 6'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_100;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_init_pp0_iter1_reg == 1'b1))) begin
        ap_sig_allocacmp_t_4 = 4'd0;
    end else begin
        ap_sig_allocacmp_t_4 = t_fu_92;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_init_pp0_iter1_reg == 1'b1))) begin
        ap_sig_allocacmp_ug_load = 2'd0;
    end else begin
        ap_sig_allocacmp_ug_load = ug_fu_96;
    end
end

always @ (*) begin
    if (((ap_predicate_op57_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        g_stream_blk_n = g_stream_full_n;
    end else begin
        g_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_predicate_op57_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        g_stream_write = 1'b1;
    end else begin
        g_stream_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (ap_start_int == 1'b0)) & (icmp_ln58_fu_154_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        mlp_i_stream_TDATA_blk_n = mlp_i_stream_TVALID;
    end else begin
        mlp_i_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (icmp_ln58_fu_154_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        mlp_i_stream_TREADY = 1'b1;
    end else begin
        mlp_i_stream_TREADY = 1'b0;
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
    if (((start_once_reg == 1'b0) & (real_start == 1'b1))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
    end
end

always @ (*) begin
    if (((ap_predicate_op59_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        u_stream_blk_n = u_stream_full_n;
    end else begin
        u_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_predicate_op59_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        u_stream_write = 1'b1;
    end else begin
        u_stream_write = 1'b0;
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if (((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (icmp_ln58_reg_389_pp0_iter1_reg == 1'd1) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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

assign add_ln58_fu_160_p2 = (ap_sig_allocacmp_indvar_flatten11_load + 13'd1);

assign add_ln59_1_fu_246_p2 = (ap_sig_allocacmp_indvar_flatten_load + 6'd1);

assign add_ln60_fu_332_p2 = (ap_sig_allocacmp_t_4 + 4'd1);

assign and_ln58_fu_294_p2 = (xor_ln58_fu_283_p2 & icmp_ln60_fu_288_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_done_reg == 1'b1) | (ap_start_int == 1'b0) | ((icmp_ln58_fu_154_p2 == 1'd0) & (mlp_i_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state3_pp0_stage0_iter2 = (((ap_predicate_op59_write_state3 == 1'b1) & (u_stream_full_n == 1'b0)) | ((ap_predicate_op57_write_state3 == 1'b1) & (g_stream_full_n == 1'b0)));
end

always @ (*) begin
    ap_condition_107 = (~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_141 = (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op57_write_state3 = ((cmp15_reg_440 == 1'd0) & (icmp_ln58_reg_389_pp0_iter1_reg == 1'd0));
end

always @ (*) begin
    ap_predicate_op59_write_state3 = ((cmp15_reg_440 == 1'd1) & (icmp_ln58_reg_389_pp0_iter1_reg == 1'd0));
end

assign ap_ready = internal_ap_ready;

assign cmp15_fu_314_p2 = ((select_ln59_fu_306_p3 == 2'd0) ? 1'b1 : 1'b0);

assign g_stream_din = or_ln71_s_reg_444;

assign icmp_ln58_fu_154_p2 = ((ap_sig_allocacmp_indvar_flatten11_load == 13'd5120) ? 1'b1 : 1'b0);

assign icmp_ln58_reg_389_pp0_iter0_reg = icmp_ln58_reg_389;

assign icmp_ln59_fu_166_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 6'd16) ? 1'b1 : 1'b0);

assign icmp_ln60_fu_288_p2 = ((ap_sig_allocacmp_t_4 == 4'd8) ? 1'b1 : 1'b0);

assign or_ln60_fu_338_p2 = (icmp_ln59_reg_393 | and_ln58_fu_294_p2);

assign or_ln71_s_fu_320_p9 = {{{{{{{{tmp_7_reg_435}, {tmp_6_reg_430}}, {tmp_5_reg_425}}, {tmp_4_reg_420}}, {tmp_3_reg_415}}, {tmp_2_reg_410}}, {tmp_reg_405}}, {out_reg_400}};

assign out_fu_172_p1 = mlp_i_stream_TDATA[20:0];

assign select_ln58_fu_276_p3 = ((icmp_ln59_reg_393[0:0] == 1'b1) ? 2'd0 : ap_sig_allocacmp_ug_load);

assign select_ln59_1_fu_252_p3 = ((icmp_ln59_fu_166_p2[0:0] == 1'b1) ? 6'd1 : add_ln59_1_fu_246_p2);

assign select_ln59_fu_306_p3 = ((and_ln58_fu_294_p2[0:0] == 1'b1) ? ug_2_fu_300_p2 : select_ln58_fu_276_p3);

assign start_out = real_start;

assign t_5_fu_343_p3 = ((or_ln60_fu_338_p2[0:0] == 1'b1) ? 4'd1 : add_ln60_fu_332_p2);

assign u_stream_din = or_ln71_s_reg_444;

assign ug_2_fu_300_p2 = (select_ln58_fu_276_p3 + 2'd1);

assign xor_ln58_fu_283_p2 = (icmp_ln59_reg_393 ^ 1'd1);

endmodule //SILU_GELU_llm_split_ug
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_llm_merge_silu_u (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        u_stream_dout,
        u_stream_num_data_valid,
        u_stream_fifo_cap,
        u_stream_empty_n,
        u_stream_read,
        silu_stream_dout,
        silu_stream_num_data_valid,
        silu_stream_fifo_cap,
        silu_stream_empty_n,
        silu_stream_read,
        act_stream_din,
        act_stream_num_data_valid,
        act_stream_fifo_cap,
        act_stream_full_n,
        act_stream_write
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter4_fsm_state5 = 2'd2;
parameter    ap_ST_iter5_fsm_state6 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [167:0] u_stream_dout;
input  [4:0] u_stream_num_data_valid;
input  [4:0] u_stream_fifo_cap;
input   u_stream_empty_n;
output   u_stream_read;
input  [167:0] silu_stream_dout;
input  [4:0] silu_stream_num_data_valid;
input  [4:0] silu_stream_fifo_cap;
input   silu_stream_empty_n;
output   silu_stream_read;
output  [215:0] act_stream_din;
input  [5:0] act_stream_num_data_valid;
input  [5:0] act_stream_fifo_cap;
input   act_stream_full_n;
output   act_stream_write;

reg ap_idle;
reg u_stream_read;
reg silu_stream_read;
reg act_stream_write;

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
wire   [0:0] icmp_ln84_fu_165_p2;
reg    ap_done_reg;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
reg   [0:0] icmp_ln84_reg_494;
reg   [0:0] icmp_ln84_reg_494_pp0_iter4_reg;
reg    ap_block_state6_pp0_stage0_iter5;
wire    ap_CS_iter5_fsm_state6;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    u_stream_blk_n;
reg    silu_stream_blk_n;
reg    act_stream_blk_n;
reg   [0:0] icmp_ln84_reg_494_pp0_iter1_reg;
reg   [0:0] icmp_ln84_reg_494_pp0_iter2_reg;
reg   [0:0] icmp_ln84_reg_494_pp0_iter3_reg;
wire   [20:0] trunc_ln87_fu_177_p1;
reg  signed [20:0] trunc_ln87_reg_498;
reg  signed [20:0] trunc_ln87_1_reg_503;
reg  signed [20:0] trunc_ln87_2_reg_508;
reg  signed [20:0] trunc_ln87_3_reg_513;
reg  signed [20:0] trunc_ln87_4_reg_518;
reg  signed [20:0] trunc_ln87_5_reg_523;
reg  signed [20:0] trunc_ln87_6_reg_528;
reg  signed [20:0] trunc_ln87_7_reg_533;
wire   [20:0] trunc_ln88_fu_251_p1;
reg  signed [20:0] trunc_ln88_reg_538;
reg  signed [20:0] trunc_ln88_1_reg_543;
reg  signed [20:0] trunc_ln88_2_reg_548;
reg  signed [20:0] trunc_ln88_3_reg_553;
reg  signed [20:0] trunc_ln88_4_reg_558;
reg  signed [20:0] trunc_ln88_5_reg_563;
reg  signed [20:0] trunc_ln88_6_reg_568;
reg  signed [20:0] trunc_ln88_7_reg_573;
reg   [26:0] xm_val_reg_658;
reg   [26:0] xm_val_1_reg_663;
reg   [26:0] xm_val_2_reg_668;
reg   [26:0] xm_val_3_reg_673;
reg   [26:0] xm_val_4_reg_678;
reg   [26:0] xm_val_5_reg_683;
reg   [26:0] xm_val_6_reg_688;
reg   [26:0] xm_val_7_reg_693;
reg   [11:0] indvar_flatten_fu_102;
wire   [11:0] add_ln84_fu_171_p2;
wire    ap_loop_init;
reg   [11:0] ap_sig_allocacmp_indvar_flatten_load;
wire   [41:0] grp_fu_125_p2;
wire   [41:0] grp_fu_129_p2;
wire   [41:0] grp_fu_133_p2;
wire   [41:0] grp_fu_137_p2;
wire   [41:0] grp_fu_141_p2;
wire   [41:0] grp_fu_145_p2;
wire   [41:0] grp_fu_149_p2;
wire   [41:0] grp_fu_153_p2;
reg    grp_fu_125_ce;
reg    grp_fu_129_ce;
reg    grp_fu_133_ce;
reg    grp_fu_137_ce;
reg    grp_fu_141_ce;
reg    grp_fu_145_ce;
reg    grp_fu_149_ce;
reg    grp_fu_153_ce;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
reg    ap_ST_iter5_fsm_state6_blk;
wire    ap_start_int;
reg    ap_condition_136;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 ap_CS_iter4_fsm = 2'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 ap_done_reg = 1'b0;
//#0 indvar_flatten_fu_102 = 12'd0;
end

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U21(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_reg_538),
    .din1(trunc_ln87_reg_498),
    .ce(grp_fu_125_ce),
    .dout(grp_fu_125_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U22(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_1_reg_543),
    .din1(trunc_ln87_1_reg_503),
    .ce(grp_fu_129_ce),
    .dout(grp_fu_129_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U23(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_2_reg_548),
    .din1(trunc_ln87_2_reg_508),
    .ce(grp_fu_133_ce),
    .dout(grp_fu_133_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U24(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_3_reg_553),
    .din1(trunc_ln87_3_reg_513),
    .ce(grp_fu_137_ce),
    .dout(grp_fu_137_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U25(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_4_reg_558),
    .din1(trunc_ln87_4_reg_518),
    .ce(grp_fu_141_ce),
    .dout(grp_fu_141_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U26(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_5_reg_563),
    .din1(trunc_ln87_5_reg_523),
    .ce(grp_fu_145_ce),
    .dout(grp_fu_145_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U27(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_6_reg_568),
    .din1(trunc_ln87_6_reg_528),
    .ce(grp_fu_149_ce),
    .dout(grp_fu_149_p2)
);

SILU_GELU_mul_21s_21s_42_4_1 #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 21 ),
    .din1_WIDTH( 21 ),
    .dout_WIDTH( 42 ))
mul_21s_21s_42_4_1_U28(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(trunc_ln88_7_reg_573),
    .din1(trunc_ln87_7_reg_533),
    .ce(grp_fu_153_ce),
    .dout(grp_fu_153_p2)
);

SILU_GELU_flow_control_loop_pipe flow_control_loop_pipe_U(
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
    .ap_done_int(ap_done_int),
    .ap_continue(ap_continue)
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
        end else if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_136)) begin
        if ((icmp_ln84_fu_165_p2 == 1'd0)) begin
            indvar_flatten_fu_102 <= add_ln84_fu_171_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_102 <= 12'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln84_reg_494 <= icmp_ln84_fu_165_p2;
        trunc_ln87_1_reg_503 <= {{u_stream_dout[41:21]}};
        trunc_ln87_2_reg_508 <= {{u_stream_dout[62:42]}};
        trunc_ln87_3_reg_513 <= {{u_stream_dout[83:63]}};
        trunc_ln87_4_reg_518 <= {{u_stream_dout[104:84]}};
        trunc_ln87_5_reg_523 <= {{u_stream_dout[125:105]}};
        trunc_ln87_6_reg_528 <= {{u_stream_dout[146:126]}};
        trunc_ln87_7_reg_533 <= {{u_stream_dout[167:147]}};
        trunc_ln87_reg_498 <= trunc_ln87_fu_177_p1;
        trunc_ln88_1_reg_543 <= {{silu_stream_dout[41:21]}};
        trunc_ln88_2_reg_548 <= {{silu_stream_dout[62:42]}};
        trunc_ln88_3_reg_553 <= {{silu_stream_dout[83:63]}};
        trunc_ln88_4_reg_558 <= {{silu_stream_dout[104:84]}};
        trunc_ln88_5_reg_563 <= {{silu_stream_dout[125:105]}};
        trunc_ln88_6_reg_568 <= {{silu_stream_dout[146:126]}};
        trunc_ln88_7_reg_573 <= {{silu_stream_dout[167:147]}};
        trunc_ln88_reg_538 <= trunc_ln88_fu_251_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln84_reg_494_pp0_iter1_reg <= icmp_ln84_reg_494;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln84_reg_494_pp0_iter2_reg <= icmp_ln84_reg_494_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln84_reg_494_pp0_iter3_reg <= icmp_ln84_reg_494_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        icmp_ln84_reg_494_pp0_iter4_reg <= icmp_ln84_reg_494_pp0_iter3_reg;
        xm_val_1_reg_663 <= {{grp_fu_129_p2[38:12]}};
        xm_val_2_reg_668 <= {{grp_fu_133_p2[38:12]}};
        xm_val_3_reg_673 <= {{grp_fu_137_p2[38:12]}};
        xm_val_4_reg_678 <= {{grp_fu_141_p2[38:12]}};
        xm_val_5_reg_683 <= {{grp_fu_145_p2[38:12]}};
        xm_val_6_reg_688 <= {{grp_fu_149_p2[38:12]}};
        xm_val_7_reg_693 <= {{grp_fu_153_p2[38:12]}};
        xm_val_reg_658 <= {{grp_fu_125_p2[38:12]}};
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln84_reg_494_pp0_iter4_reg == 1'd0))) begin
        act_stream_blk_n = act_stream_full_n;
    end else begin
        act_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln84_reg_494_pp0_iter4_reg == 1'd0))) begin
        act_stream_write = 1'b1;
    end else begin
        act_stream_write = 1'b0;
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
    if ((1'b1 == ap_block_state6_pp0_stage0_iter5)) begin
        ap_ST_iter5_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter5_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (icmp_ln84_fu_165_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1))) begin
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
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 12'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_102;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_125_ce = 1'b1;
    end else begin
        grp_fu_125_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_129_ce = 1'b1;
    end else begin
        grp_fu_129_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_133_ce = 1'b1;
    end else begin
        grp_fu_133_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_137_ce = 1'b1;
    end else begin
        grp_fu_137_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_141_ce = 1'b1;
    end else begin
        grp_fu_141_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_145_ce = 1'b1;
    end else begin
        grp_fu_145_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_149_ce = 1'b1;
    end else begin
        grp_fu_149_ce = 1'b0;
    end
end

always @ (*) begin
    if (((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4)) | (~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        grp_fu_153_ce = 1'b1;
    end else begin
        grp_fu_153_ce = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (ap_start_int == 1'b0)) & (icmp_ln84_fu_165_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        silu_stream_blk_n = silu_stream_empty_n;
    end else begin
        silu_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (icmp_ln84_fu_165_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        silu_stream_read = 1'b1;
    end else begin
        silu_stream_read = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (ap_start_int == 1'b0)) & (icmp_ln84_fu_165_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        u_stream_blk_n = u_stream_empty_n;
    end else begin
        u_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (icmp_ln84_fu_165_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        u_stream_read = 1'b1;
    end else begin
        u_stream_read = 1'b0;
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b0 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
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
            if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else if (((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln84_reg_494_pp0_iter4_reg == 1'd1)) | (~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state5)))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
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

assign act_stream_din = {{{{{{{{xm_val_7_reg_693}, {xm_val_6_reg_688}}, {xm_val_5_reg_683}}, {xm_val_4_reg_678}}, {xm_val_3_reg_673}}, {xm_val_2_reg_668}}, {xm_val_1_reg_663}}, {xm_val_reg_658}};

assign add_ln84_fu_171_p2 = (ap_sig_allocacmp_indvar_flatten_load + 12'd1);

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

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_done_reg == 1'b1) | (ap_start_int == 1'b0) | ((silu_stream_empty_n == 1'b0) & (icmp_ln84_fu_165_p2 == 1'd0)) | ((icmp_ln84_fu_165_p2 == 1'd0) & (u_stream_empty_n == 1'b0)));
end

always @ (*) begin
    ap_block_state6_pp0_stage0_iter5 = ((1'b0 == act_stream_full_n) & (icmp_ln84_reg_494_pp0_iter4_reg == 1'd0));
end

always @ (*) begin
    ap_condition_136 = (~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln84_fu_165_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 12'd2560) ? 1'b1 : 1'b0);

assign trunc_ln87_fu_177_p1 = u_stream_dout[20:0];

assign trunc_ln88_fu_251_p1 = silu_stream_dout[20:0];

endmodule //SILU_GELU_llm_merge_silu_u
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module SILU_GELU_start_for_do_silu_parallel_U0
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
    SILU_GELU_start_for_do_silu_parallel_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_start_for_do_silu_parallel_U0_ShiftReg (
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


module SILU_GELU_start_for_do_silu_parallel_U0_ShiftReg
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
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module SILU_GELU_start_for_llm_merge_silu_u_U0
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 1,
    ADDR_WIDTH   = 2,
    DEPTH        = 3)
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
    SILU_GELU_start_for_llm_merge_silu_u_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_start_for_llm_merge_silu_u_U0_ShiftReg (
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


module SILU_GELU_start_for_llm_merge_silu_u_U0_ShiftReg
#(parameter
    DATA_WIDTH  = 1,
    ADDR_WIDTH  = 2,
    DEPTH       = 3)
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

(* CORE_GENERATION_INFO="SILU_GELU_SILU_GELU,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xck26-sfvc784-2LV-c,HLS_INPUT_CLOCK=2.500000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=2.412000,HLS_SYN_LAT=-1,HLS_SYN_TPT=none,HLS_SYN_MEM=51,HLS_SYN_DSP=0,HLS_SYN_FF=10950,HLS_SYN_LUT=8263,HLS_VERSION=2023_2}" *)

module SILU_GELU (
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
        mlp_i_stream_TDATA,
        mlp_i_stream_TVALID,
        mlp_i_stream_TREADY,
        q_stream_TDATA,
        q_stream_TVALID,
        q_stream_TREADY,
        s_stream_TDATA,
        s_stream_TVALID,
        s_stream_TREADY
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
input   ap_rst_n;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [0:0] mode;
input  [31:0] l_begin;
input  [31:0] l_close;
input  [223:0] mlp_i_stream_TDATA;
input   mlp_i_stream_TVALID;
output   mlp_i_stream_TREADY;
output  [63:0] q_stream_TDATA;
output   q_stream_TVALID;
input   q_stream_TREADY;
output  [7:0] s_stream_TDATA;
output   s_stream_TVALID;
input   s_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [7:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_block_state1;
wire   [31:0] add_ln236_fu_107_p2;
reg   [31:0] add_ln236_reg_223;
wire   [0:0] empty_fu_113_p2;
reg   [0:0] empty_reg_230;
wire   [0:0] empty_43_fu_119_p2;
reg   [0:0] empty_43_reg_235;
wire   [31:0] sub_ln236_fu_151_p2;
reg   [31:0] sub_ln236_reg_240;
wire    ap_CS_fsm_state2;
wire   [31:0] sub_ln236_1_fu_161_p2;
reg   [31:0] sub_ln236_1_reg_246;
wire   [0:0] empty_44_fu_166_p2;
reg   [0:0] empty_44_reg_252;
wire    ap_CS_fsm_state3;
wire   [31:0] sub_ln236_2_fu_175_p2;
reg   [31:0] sub_ln236_2_reg_257;
wire    ap_CS_fsm_state4;
wire   [63:0] grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TDATA;
wire   [7:0] grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TDATA;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_mlp_i_stream_TREADY;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TVALID;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TREADY;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TVALID;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TREADY;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready;
wire    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_idle;
reg    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_continue;
reg    grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start_reg;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state7;
wire    ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready;
wire    ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done;
reg    ap_block_state7_on_subcall_done;
reg    ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready;
reg    ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done;
reg   [31:0] l_fu_60;
wire   [31:0] add_ln236_1_fu_188_p2;
wire    ap_CS_fsm_state5;
wire   [0:0] icmp_ln236_fu_183_p2;
wire   [31:0] select_ln236_1_fu_99_p3;
wire   [30:0] select_ln236_fu_130_p3;
wire   [30:0] trunc_ln210_fu_137_p1;
wire   [30:0] smax_fu_140_p3;
wire   [31:0] zext_ln236_fu_147_p1;
wire   [31:0] smax1_fu_156_p3;
wire   [31:0] umax_fu_170_p3;
wire    ap_CS_fsm_state8;
wire    regslice_both_q_stream_U_apdone_blk;
wire    regslice_both_s_stream_U_apdone_blk;
reg    ap_block_state8;
reg   [7:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
reg    ap_ST_fsm_state7_blk;
reg    ap_ST_fsm_state8_blk;
wire    regslice_both_mlp_i_stream_U_apdone_blk;
wire   [223:0] mlp_i_stream_TDATA_int_regslice;
wire    mlp_i_stream_TVALID_int_regslice;
reg    mlp_i_stream_TREADY_int_regslice;
wire    regslice_both_mlp_i_stream_U_ack_in;
wire    q_stream_TREADY_int_regslice;
wire    regslice_both_q_stream_U_vld_out;
wire    s_stream_TREADY_int_regslice;
wire    regslice_both_s_stream_U_vld_out;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 8'd1;
//#0 grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start_reg = 1'b0;
//#0 ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready = 1'b0;
//#0 ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done = 1'b0;
//#0 l_fu_60 = 32'd0;
end

SILU_GELU_dataflow_in_loop_VITIS_LOOP_236_1_1 grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82(
    .mode(mode),
    .mlp_i_stream_TDATA(mlp_i_stream_TDATA_int_regslice),
    .q_stream_TDATA(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TDATA),
    .s_stream_TDATA(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TDATA),
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .mode_ap_vld(1'b1),
    .mlp_i_stream_TVALID(mlp_i_stream_TVALID_int_regslice),
    .mlp_i_stream_TREADY(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_mlp_i_stream_TREADY),
    .ap_start(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start),
    .q_stream_TVALID(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TVALID),
    .q_stream_TREADY(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TREADY),
    .s_stream_TVALID(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TVALID),
    .s_stream_TREADY(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TREADY),
    .ap_done(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done),
    .ap_ready(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready),
    .ap_idle(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_idle),
    .ap_continue(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_continue)
);

SILU_GELU_regslice_both #(
    .DataWidth( 224 ))
regslice_both_mlp_i_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(mlp_i_stream_TDATA),
    .vld_in(mlp_i_stream_TVALID),
    .ack_in(regslice_both_mlp_i_stream_U_ack_in),
    .data_out(mlp_i_stream_TDATA_int_regslice),
    .vld_out(mlp_i_stream_TVALID_int_regslice),
    .ack_out(mlp_i_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_mlp_i_stream_U_apdone_blk)
);

SILU_GELU_regslice_both #(
    .DataWidth( 64 ))
regslice_both_q_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TDATA),
    .vld_in(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TVALID),
    .ack_in(q_stream_TREADY_int_regslice),
    .data_out(q_stream_TDATA),
    .vld_out(regslice_both_q_stream_U_vld_out),
    .ack_out(q_stream_TREADY),
    .apdone_blk(regslice_both_q_stream_U_apdone_blk)
);

SILU_GELU_regslice_both #(
    .DataWidth( 8 ))
regslice_both_s_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TDATA),
    .vld_in(grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TVALID),
    .ack_in(s_stream_TREADY_int_regslice),
    .data_out(s_stream_TDATA),
    .vld_out(regslice_both_s_stream_U_vld_out),
    .ack_out(s_stream_TREADY),
    .apdone_blk(regslice_both_s_stream_U_apdone_blk)
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
        end else if (((1'b0 == ap_block_state8) & (1'b1 == ap_CS_fsm_state8))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state7_on_subcall_done) & (1'b1 == ap_CS_fsm_state7))) begin
            ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done <= 1'b0;
        end else if ((grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done == 1'b1)) begin
            ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state7_on_subcall_done) & (1'b1 == ap_CS_fsm_state7))) begin
            ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready <= 1'b0;
        end else if ((grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready == 1'b1)) begin
            ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state6) | ((1'b1 == ap_CS_fsm_state7) & (ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready == 1'b0)))) begin
            grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start_reg <= 1'b1;
        end else if ((grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready == 1'b1)) begin
            grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        l_fu_60 <= l_begin;
    end else if (((icmp_ln236_fu_183_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state5))) begin
        l_fu_60 <= add_ln236_1_fu_188_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        add_ln236_reg_223 <= add_ln236_fu_107_p2;
        empty_43_reg_235 <= empty_43_fu_119_p2;
        empty_reg_230 <= empty_fu_113_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        empty_44_reg_252 <= empty_44_fu_166_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        sub_ln236_1_reg_246 <= sub_ln236_1_fu_161_p2;
        sub_ln236_reg_240 <= sub_ln236_fu_151_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        sub_ln236_2_reg_257 <= sub_ln236_2_fu_175_p2;
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

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state7_on_subcall_done)) begin
        ap_ST_fsm_state7_blk = 1'b1;
    end else begin
        ap_ST_fsm_state7_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state8)) begin
        ap_ST_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_fsm_state8_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state8) & (1'b1 == ap_CS_fsm_state8))) begin
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
    if (((1'b0 == ap_block_state8) & (1'b1 == ap_CS_fsm_state8))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state7_on_subcall_done) & (1'b1 == ap_CS_fsm_state7))) begin
        grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_continue = 1'b1;
    end else begin
        grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_continue = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        mlp_i_stream_TREADY_int_regslice = grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_mlp_i_stream_TREADY;
    end else begin
        mlp_i_stream_TREADY_int_regslice = 1'b0;
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
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((icmp_ln236_fu_183_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state5))) begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            if (((1'b0 == ap_block_state7_on_subcall_done) & (1'b1 == ap_CS_fsm_state7))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state8 : begin
            if (((1'b0 == ap_block_state8) & (1'b1 == ap_CS_fsm_state8))) begin
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

assign add_ln236_1_fu_188_p2 = (l_fu_60 + 32'd1);

assign add_ln236_fu_107_p2 = ($signed(l_begin) + $signed(32'd4294967295));

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

always @ (*) begin
    ap_block_state1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0));
end

always @ (*) begin
    ap_block_state7_on_subcall_done = ((ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready & ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done) == 1'b0);
end

always @ (*) begin
    ap_block_state8 = ((regslice_both_s_stream_U_apdone_blk == 1'b1) | (regslice_both_q_stream_U_apdone_blk == 1'b1));
end

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done = (grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done | ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_done);

assign ap_sync_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready = (grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready | ap_sync_reg_grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_ready);

assign empty_43_fu_119_p2 = (($signed(l_close) > $signed(l_begin)) ? 1'b1 : 1'b0);

assign empty_44_fu_166_p2 = ((sub_ln236_reg_240 > sub_ln236_1_reg_246) ? 1'b1 : 1'b0);

assign empty_fu_113_p2 = (($signed(select_ln236_1_fu_99_p3) > $signed(l_begin)) ? 1'b1 : 1'b0);

assign grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start = grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_ap_start_reg;

assign grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_q_stream_TREADY = (q_stream_TREADY_int_regslice & ap_CS_fsm_state7);

assign grp_dataflow_in_loop_VITIS_LOOP_236_1_1_fu_82_s_stream_TREADY = (s_stream_TREADY_int_regslice & ap_CS_fsm_state7);

assign icmp_ln236_fu_183_p2 = ((l_fu_60 == sub_ln236_2_reg_257) ? 1'b1 : 1'b0);

assign mlp_i_stream_TREADY = regslice_both_mlp_i_stream_U_ack_in;

assign q_stream_TVALID = regslice_both_q_stream_U_vld_out;

assign s_stream_TVALID = regslice_both_s_stream_U_vld_out;

assign select_ln236_1_fu_99_p3 = ((mode[0:0] == 1'b1) ? 32'd12 : 32'd32);

assign select_ln236_fu_130_p3 = ((mode[0:0] == 1'b1) ? 31'd12 : 31'd32);

assign smax1_fu_156_p3 = ((empty_43_reg_235[0:0] == 1'b1) ? l_close : l_begin);

assign smax_fu_140_p3 = ((empty_reg_230[0:0] == 1'b1) ? select_ln236_fu_130_p3 : trunc_ln210_fu_137_p1);

assign sub_ln236_1_fu_161_p2 = (add_ln236_reg_223 - smax1_fu_156_p3);

assign sub_ln236_2_fu_175_p2 = (add_ln236_reg_223 - umax_fu_170_p3);

assign sub_ln236_fu_151_p2 = (add_ln236_reg_223 - zext_ln236_fu_147_p1);

assign trunc_ln210_fu_137_p1 = l_begin[30:0];

assign umax_fu_170_p3 = ((empty_44_reg_252[0:0] == 1'b1) ? sub_ln236_reg_240 : sub_ln236_1_reg_246);

assign zext_ln236_fu_147_p1 = smax_fu_140_p3;

endmodule //SILU_GELU
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module SILU_GELU_fifo_w1_d2_S
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 1,
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
    SILU_GELU_fifo_w1_d2_S_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_fifo_w1_d2_S_ShiftReg (
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


module SILU_GELU_fifo_w1_d2_S_ShiftReg
#(parameter
    DATA_WIDTH  = 1,
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
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 2 

module SILU_GELU_fifo_w216_d32_A
#(parameter
    MEM_STYLE    = "auto",
    DATA_WIDTH   = 216,
    ADDR_WIDTH   = 5,
    DEPTH        = 31)
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
    SILU_GELU_fifo_w216_d32_A_ram 
    #(  .MEM_STYLE  (MEM_STYLE),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH)
    ) U_SILU_GELU_fifo_w216_d32_A_ram (
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


module SILU_GELU_fifo_w216_d32_A_ram
#(parameter
    MEM_STYLE   = "auto",
    DATA_WIDTH  = 216,
    ADDR_WIDTH  = 5,
    DEPTH       = 31)
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

`timescale 1 ns / 1 ps 

module SILU_GELU_do_silu (
        g_stream_dout,
        g_stream_empty_n,
        g_stream_read,
        silu_stream_din,
        silu_stream_full_n,
        silu_stream_write,
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_ready,
        ap_idle,
        ap_continue
);


input  [167:0] g_stream_dout;
input   g_stream_empty_n;
output   g_stream_read;
output  [167:0] silu_stream_din;
input   silu_stream_full_n;
output   silu_stream_write;
input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_ready;
output   ap_idle;
input   ap_continue;

wire    do_adapt_U0_ap_start;
wire    do_adapt_U0_ap_done;
wire    do_adapt_U0_ap_continue;
wire    do_adapt_U0_ap_idle;
wire    do_adapt_U0_ap_ready;
wire    do_adapt_U0_start_out;
wire    do_adapt_U0_start_write;
wire    do_adapt_U0_g_stream_read;
wire   [20:0] do_adapt_U0_adpt_stream_din;
wire    do_adapt_U0_adpt_stream_write;
wire    do_silu_parallel_U0_ap_start;
wire    do_silu_parallel_U0_ap_done;
wire    do_silu_parallel_U0_ap_continue;
wire    do_silu_parallel_U0_ap_idle;
wire    do_silu_parallel_U0_ap_ready;
wire    do_silu_parallel_U0_adpt_stream_read;
wire   [20:0] do_silu_parallel_U0_silu_stream_din;
wire    do_silu_parallel_U0_silu_stream_write;
wire    do_silu_parallel_U0_start_out;
wire    do_silu_parallel_U0_start_write;
wire    do_adapt_1_U0_ap_start;
wire    do_adapt_1_U0_ap_done;
wire    do_adapt_1_U0_ap_continue;
wire    do_adapt_1_U0_ap_idle;
wire    do_adapt_1_U0_ap_ready;
wire    do_adapt_1_U0_silu_stream_read;
wire   [167:0] do_adapt_1_U0_silu_stream1_din;
wire    do_adapt_1_U0_silu_stream1_write;
wire    adpt_stream_full_n;
wire   [20:0] adpt_stream_dout;
wire   [2:0] adpt_stream_num_data_valid;
wire   [2:0] adpt_stream_fifo_cap;
wire    adpt_stream_empty_n;
wire    silu_stream1_full_n;
wire   [20:0] silu_stream1_dout;
wire   [2:0] silu_stream1_num_data_valid;
wire   [2:0] silu_stream1_fifo_cap;
wire    silu_stream1_empty_n;
wire   [0:0] start_for_do_silu_parallel_U0_din;
wire    start_for_do_silu_parallel_U0_full_n;
wire   [0:0] start_for_do_silu_parallel_U0_dout;
wire    start_for_do_silu_parallel_U0_empty_n;
wire   [0:0] start_for_do_adapt_1_U0_din;
wire    start_for_do_adapt_1_U0_full_n;
wire   [0:0] start_for_do_adapt_1_U0_dout;
wire    start_for_do_adapt_1_U0_empty_n;

SILU_GELU_do_adapt do_adapt_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(do_adapt_U0_ap_start),
    .start_full_n(start_for_do_silu_parallel_U0_full_n),
    .ap_done(do_adapt_U0_ap_done),
    .ap_continue(do_adapt_U0_ap_continue),
    .ap_idle(do_adapt_U0_ap_idle),
    .ap_ready(do_adapt_U0_ap_ready),
    .start_out(do_adapt_U0_start_out),
    .start_write(do_adapt_U0_start_write),
    .g_stream_dout(g_stream_dout),
    .g_stream_num_data_valid(5'd0),
    .g_stream_fifo_cap(5'd0),
    .g_stream_empty_n(g_stream_empty_n),
    .g_stream_read(do_adapt_U0_g_stream_read),
    .adpt_stream_din(do_adapt_U0_adpt_stream_din),
    .adpt_stream_num_data_valid(adpt_stream_num_data_valid),
    .adpt_stream_fifo_cap(adpt_stream_fifo_cap),
    .adpt_stream_full_n(adpt_stream_full_n),
    .adpt_stream_write(do_adapt_U0_adpt_stream_write)
);

SILU_GELU_do_silu_parallel do_silu_parallel_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(do_silu_parallel_U0_ap_start),
    .start_full_n(start_for_do_adapt_1_U0_full_n),
    .ap_done(do_silu_parallel_U0_ap_done),
    .ap_continue(do_silu_parallel_U0_ap_continue),
    .ap_idle(do_silu_parallel_U0_ap_idle),
    .ap_ready(do_silu_parallel_U0_ap_ready),
    .adpt_stream_dout(adpt_stream_dout),
    .adpt_stream_num_data_valid(adpt_stream_num_data_valid),
    .adpt_stream_fifo_cap(adpt_stream_fifo_cap),
    .adpt_stream_empty_n(adpt_stream_empty_n),
    .adpt_stream_read(do_silu_parallel_U0_adpt_stream_read),
    .silu_stream_din(do_silu_parallel_U0_silu_stream_din),
    .silu_stream_num_data_valid(silu_stream1_num_data_valid),
    .silu_stream_fifo_cap(silu_stream1_fifo_cap),
    .silu_stream_full_n(silu_stream1_full_n),
    .silu_stream_write(do_silu_parallel_U0_silu_stream_write),
    .start_out(do_silu_parallel_U0_start_out),
    .start_write(do_silu_parallel_U0_start_write)
);

SILU_GELU_do_adapt_1 do_adapt_1_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(do_adapt_1_U0_ap_start),
    .ap_done(do_adapt_1_U0_ap_done),
    .ap_continue(do_adapt_1_U0_ap_continue),
    .ap_idle(do_adapt_1_U0_ap_idle),
    .ap_ready(do_adapt_1_U0_ap_ready),
    .silu_stream_dout(silu_stream1_dout),
    .silu_stream_num_data_valid(silu_stream1_num_data_valid),
    .silu_stream_fifo_cap(silu_stream1_fifo_cap),
    .silu_stream_empty_n(silu_stream1_empty_n),
    .silu_stream_read(do_adapt_1_U0_silu_stream_read),
    .silu_stream1_din(do_adapt_1_U0_silu_stream1_din),
    .silu_stream1_num_data_valid(5'd0),
    .silu_stream1_fifo_cap(5'd0),
    .silu_stream1_full_n(silu_stream_full_n),
    .silu_stream1_write(do_adapt_1_U0_silu_stream1_write)
);

SILU_GELU_fifo_w21_d2_S adpt_stream_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(do_adapt_U0_adpt_stream_din),
    .if_full_n(adpt_stream_full_n),
    .if_write(do_adapt_U0_adpt_stream_write),
    .if_dout(adpt_stream_dout),
    .if_num_data_valid(adpt_stream_num_data_valid),
    .if_fifo_cap(adpt_stream_fifo_cap),
    .if_empty_n(adpt_stream_empty_n),
    .if_read(do_silu_parallel_U0_adpt_stream_read)
);

SILU_GELU_fifo_w21_d2_S silu_stream1_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(do_silu_parallel_U0_silu_stream_din),
    .if_full_n(silu_stream1_full_n),
    .if_write(do_silu_parallel_U0_silu_stream_write),
    .if_dout(silu_stream1_dout),
    .if_num_data_valid(silu_stream1_num_data_valid),
    .if_fifo_cap(silu_stream1_fifo_cap),
    .if_empty_n(silu_stream1_empty_n),
    .if_read(do_adapt_1_U0_silu_stream_read)
);

SILU_GELU_start_for_do_silu_parallel_U0 start_for_do_silu_parallel_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_do_silu_parallel_U0_din),
    .if_full_n(start_for_do_silu_parallel_U0_full_n),
    .if_write(do_adapt_U0_start_write),
    .if_dout(start_for_do_silu_parallel_U0_dout),
    .if_empty_n(start_for_do_silu_parallel_U0_empty_n),
    .if_read(do_silu_parallel_U0_ap_ready)
);

SILU_GELU_start_for_do_adapt_1_U0 start_for_do_adapt_1_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_do_adapt_1_U0_din),
    .if_full_n(start_for_do_adapt_1_U0_full_n),
    .if_write(do_silu_parallel_U0_start_write),
    .if_dout(start_for_do_adapt_1_U0_dout),
    .if_empty_n(start_for_do_adapt_1_U0_empty_n),
    .if_read(do_adapt_1_U0_ap_ready)
);

assign ap_done = do_adapt_1_U0_ap_done;

assign ap_idle = (do_silu_parallel_U0_ap_idle & do_adapt_U0_ap_idle & do_adapt_1_U0_ap_idle);

assign ap_ready = do_adapt_U0_ap_ready;

assign do_adapt_1_U0_ap_continue = ap_continue;

assign do_adapt_1_U0_ap_start = start_for_do_adapt_1_U0_empty_n;

assign do_adapt_U0_ap_continue = 1'b1;

assign do_adapt_U0_ap_start = ap_start;

assign do_silu_parallel_U0_ap_continue = 1'b1;

assign do_silu_parallel_U0_ap_start = start_for_do_silu_parallel_U0_empty_n;

assign g_stream_read = do_adapt_U0_g_stream_read;

assign silu_stream_din = do_adapt_1_U0_silu_stream1_din;

assign silu_stream_write = do_adapt_1_U0_silu_stream1_write;

assign start_for_do_adapt_1_U0_din = 1'b1;

assign start_for_do_silu_parallel_U0_din = 1'b1;

endmodule //SILU_GELU_do_silu
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 2 

module SILU_GELU_fifo_w168_d16_A
#(parameter
    MEM_STYLE    = "auto",
    DATA_WIDTH   = 168,
    ADDR_WIDTH   = 4,
    DEPTH        = 15)
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
    SILU_GELU_fifo_w168_d16_A_ram 
    #(  .MEM_STYLE  (MEM_STYLE),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH)
    ) U_SILU_GELU_fifo_w168_d16_A_ram (
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


module SILU_GELU_fifo_w168_d16_A_ram
#(parameter
    MEM_STYLE   = "auto",
    DATA_WIDTH  = 168,
    ADDR_WIDTH  = 4,
    DEPTH       = 15)
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

module SILU_GELU_start_for_do_silu_U0
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
    SILU_GELU_start_for_do_silu_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_start_for_do_silu_U0_ShiftReg (
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


module SILU_GELU_start_for_do_silu_U0_ShiftReg
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
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module SILU_GELU_flow_control_loop_pipe_sequential_init(
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

// if no ap_continue port and current module is not SILU_GELU module, 
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

// if no ap_continue port and current module is not SILU_GELU module, ap_done handshakes with ap_start
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

module SILU_GELU_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        act_stream_i_dout,
        act_stream_i_num_data_valid,
        act_stream_i_fifo_cap,
        act_stream_i_empty_n,
        act_stream_i_read,
        q_stream_TREADY,
        s_stream_TREADY,
        select_ln175,
        q_stream_TDATA,
        q_stream_TVALID,
        s_stream_TDATA,
        s_stream_TVALID
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
parameter    ap_ST_iter13_fsm_state14 = 2'd2;
parameter    ap_ST_iter14_fsm_state15 = 2'd2;
parameter    ap_ST_iter15_fsm_state16 = 2'd2;
parameter    ap_ST_iter16_fsm_state17 = 2'd2;
parameter    ap_ST_iter17_fsm_state18 = 2'd2;
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
parameter    ap_ST_iter13_fsm_state0 = 2'd1;
parameter    ap_ST_iter14_fsm_state0 = 2'd1;
parameter    ap_ST_iter15_fsm_state0 = 2'd1;
parameter    ap_ST_iter16_fsm_state0 = 2'd1;
parameter    ap_ST_iter17_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [215:0] act_stream_i_dout;
input  [5:0] act_stream_i_num_data_valid;
input  [5:0] act_stream_i_fifo_cap;
input   act_stream_i_empty_n;
output   act_stream_i_read;
input   q_stream_TREADY;
input   s_stream_TREADY;
input  [17:0] select_ln175;
output  [63:0] q_stream_TDATA;
output   q_stream_TVALID;
output  [7:0] s_stream_TDATA;
output   s_stream_TVALID;

reg ap_idle;
reg act_stream_i_read;
reg q_stream_TVALID;
reg s_stream_TVALID;

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
reg   [1:0] ap_CS_iter13_fsm;
wire    ap_CS_iter13_fsm_state0;
reg   [1:0] ap_CS_iter14_fsm;
wire    ap_CS_iter14_fsm_state0;
reg   [1:0] ap_CS_iter15_fsm;
wire    ap_CS_iter15_fsm_state0;
reg   [1:0] ap_CS_iter16_fsm;
wire    ap_CS_iter16_fsm_state0;
reg   [1:0] ap_CS_iter17_fsm;
wire    ap_CS_iter17_fsm_state0;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln175_reg_1869;
reg    ap_block_state2_pp0_stage0_iter1;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
wire    ap_CS_iter5_fsm_state6;
wire    ap_CS_iter6_fsm_state7;
wire    ap_CS_iter7_fsm_state8;
wire    ap_CS_iter8_fsm_state9;
wire    ap_CS_iter9_fsm_state10;
wire    ap_CS_iter10_fsm_state11;
wire    ap_CS_iter11_fsm_state12;
wire    ap_CS_iter12_fsm_state13;
wire    ap_CS_iter13_fsm_state14;
wire    ap_CS_iter14_fsm_state15;
wire    ap_CS_iter15_fsm_state16;
wire    ap_CS_iter16_fsm_state17;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter16_reg;
reg    ap_block_state18_pp0_stage0_iter17;
reg    ap_block_state18_io;
wire    ap_CS_iter17_fsm_state18;
wire   [0:0] icmp_ln175_fu_455_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    q_stream_TDATA_blk_n;
reg    s_stream_TDATA_blk_n;
reg    act_stream_i_blk_n;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter1_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter2_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter3_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter4_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter5_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter6_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter7_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter8_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter9_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter10_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter11_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter12_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter13_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter14_reg;
reg   [0:0] icmp_ln175_reg_1869_pp0_iter15_reg;
wire   [26:0] q_val_fu_472_p1;
reg   [26:0] q_val_reg_1873;
reg   [26:0] q_val_reg_1873_pp0_iter2_reg;
reg   [26:0] q_val_reg_1873_pp0_iter3_reg;
reg   [26:0] q_val_reg_1873_pp0_iter4_reg;
reg   [26:0] q_val_reg_1873_pp0_iter5_reg;
reg   [26:0] q_val_reg_1873_pp0_iter6_reg;
reg   [26:0] q_val_reg_1873_pp0_iter7_reg;
reg   [26:0] q_val_reg_1873_pp0_iter8_reg;
reg   [26:0] q_val_reg_1873_pp0_iter9_reg;
reg   [26:0] q_val_reg_1873_pp0_iter10_reg;
reg   [26:0] q_val_reg_1873_pp0_iter11_reg;
reg   [26:0] q_val_reg_1873_pp0_iter12_reg;
reg   [26:0] q_val_reg_1873_pp0_iter13_reg;
reg   [26:0] q_val_4_reg_1882;
reg   [26:0] q_val_4_reg_1882_pp0_iter2_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter3_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter4_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter5_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter6_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter7_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter8_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter9_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter10_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter11_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter12_reg;
reg   [26:0] q_val_4_reg_1882_pp0_iter13_reg;
reg   [26:0] q_val_8_reg_1891;
reg   [26:0] q_val_8_reg_1891_pp0_iter2_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter3_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter4_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter5_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter6_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter7_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter8_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter9_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter10_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter11_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter12_reg;
reg   [26:0] q_val_8_reg_1891_pp0_iter13_reg;
reg   [26:0] q_val_16_reg_1900;
reg   [26:0] q_val_16_reg_1900_pp0_iter2_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter3_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter4_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter5_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter6_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter7_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter8_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter9_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter10_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter11_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter12_reg;
reg   [26:0] q_val_16_reg_1900_pp0_iter13_reg;
reg   [26:0] q_val_1_reg_1909;
reg   [26:0] q_val_1_reg_1909_pp0_iter2_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter3_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter4_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter5_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter6_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter7_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter8_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter9_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter10_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter11_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter12_reg;
reg   [26:0] q_val_1_reg_1909_pp0_iter13_reg;
reg   [26:0] q_val_5_reg_1918;
reg   [26:0] q_val_5_reg_1918_pp0_iter2_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter3_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter4_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter5_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter6_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter7_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter8_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter9_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter10_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter11_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter12_reg;
reg   [26:0] q_val_5_reg_1918_pp0_iter13_reg;
reg   [26:0] q_val_9_reg_1927;
reg   [26:0] q_val_9_reg_1927_pp0_iter2_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter3_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter4_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter5_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter6_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter7_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter8_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter9_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter10_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter11_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter12_reg;
reg   [26:0] q_val_9_reg_1927_pp0_iter13_reg;
reg   [26:0] q_val_2_reg_1936;
reg   [26:0] q_val_2_reg_1936_pp0_iter2_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter3_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter4_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter5_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter6_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter7_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter8_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter9_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter10_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter11_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter12_reg;
reg   [26:0] q_val_2_reg_1936_pp0_iter13_reg;
reg   [0:0] tmp_reg_1945;
reg   [0:0] tmp_reg_1945_pp0_iter2_reg;
reg   [0:0] tmp_17_reg_1950;
reg   [0:0] tmp_17_reg_1950_pp0_iter2_reg;
reg   [0:0] tmp_18_reg_1955;
reg   [0:0] tmp_18_reg_1955_pp0_iter2_reg;
reg   [0:0] tmp_18_reg_1955_pp0_iter3_reg;
reg   [0:0] tmp_19_reg_1960;
reg   [0:0] tmp_19_reg_1960_pp0_iter2_reg;
reg   [0:0] tmp_19_reg_1960_pp0_iter3_reg;
reg   [0:0] tmp_19_reg_1960_pp0_iter4_reg;
reg   [0:0] tmp_20_reg_1965;
reg   [0:0] tmp_20_reg_1965_pp0_iter2_reg;
reg   [0:0] tmp_20_reg_1965_pp0_iter3_reg;
reg   [0:0] tmp_20_reg_1965_pp0_iter4_reg;
reg   [0:0] tmp_20_reg_1965_pp0_iter5_reg;
reg   [0:0] tmp_21_reg_1970;
reg   [0:0] tmp_21_reg_1970_pp0_iter2_reg;
reg   [0:0] tmp_21_reg_1970_pp0_iter3_reg;
reg   [0:0] tmp_21_reg_1970_pp0_iter4_reg;
reg   [0:0] tmp_21_reg_1970_pp0_iter5_reg;
reg   [0:0] tmp_21_reg_1970_pp0_iter6_reg;
reg   [0:0] tmp_22_reg_1975;
reg   [0:0] tmp_22_reg_1975_pp0_iter2_reg;
reg   [0:0] tmp_22_reg_1975_pp0_iter3_reg;
reg   [0:0] tmp_22_reg_1975_pp0_iter4_reg;
reg   [0:0] tmp_22_reg_1975_pp0_iter5_reg;
reg   [0:0] tmp_22_reg_1975_pp0_iter6_reg;
reg   [0:0] tmp_22_reg_1975_pp0_iter7_reg;
reg   [0:0] tmp_23_reg_1980;
reg   [0:0] tmp_23_reg_1980_pp0_iter2_reg;
reg   [0:0] tmp_23_reg_1980_pp0_iter3_reg;
reg   [0:0] tmp_23_reg_1980_pp0_iter4_reg;
reg   [0:0] tmp_23_reg_1980_pp0_iter5_reg;
reg   [0:0] tmp_23_reg_1980_pp0_iter6_reg;
reg   [0:0] tmp_23_reg_1980_pp0_iter7_reg;
reg   [0:0] tmp_23_reg_1980_pp0_iter8_reg;
wire   [26:0] sub_ln185_fu_610_p2;
reg   [26:0] sub_ln185_reg_1985;
wire   [25:0] select_ln185_1_fu_630_p3;
reg   [25:0] select_ln185_1_reg_1990;
wire   [26:0] select_ln185_2_fu_643_p3;
reg   [26:0] select_ln185_2_reg_1995;
wire   [26:0] select_ln185_3_fu_657_p3;
reg   [26:0] select_ln185_3_reg_2001;
wire   [26:0] select_ln185_4_fu_669_p3;
reg   [26:0] select_ln185_4_reg_2007;
wire   [26:0] select_ln185_5_fu_679_p3;
reg   [26:0] select_ln185_5_reg_2013;
wire   [26:0] select_ln185_6_fu_690_p3;
reg   [26:0] select_ln185_6_reg_2019;
wire   [26:0] select_ln185_7_fu_700_p3;
reg   [26:0] select_ln185_7_reg_2025;
wire   [26:0] select_ln185_8_fu_711_p3;
reg   [26:0] select_ln185_8_reg_2031;
wire   [26:0] select_ln185_9_fu_721_p3;
reg   [26:0] select_ln185_9_reg_2037;
wire   [26:0] select_ln185_10_fu_732_p3;
reg   [26:0] select_ln185_10_reg_2043;
wire   [26:0] select_ln185_11_fu_742_p3;
reg   [26:0] select_ln185_11_reg_2049;
wire   [26:0] select_ln185_12_fu_753_p3;
reg   [26:0] select_ln185_12_reg_2055;
wire   [26:0] select_ln185_13_fu_763_p3;
reg   [26:0] select_ln185_13_reg_2061;
wire   [26:0] select_ln185_14_fu_774_p3;
reg   [26:0] select_ln185_14_reg_2067;
wire   [26:0] abs_max_fu_784_p3;
reg   [26:0] abs_max_reg_2073;
wire   [0:0] trunc_ln10_fu_807_p1;
reg   [0:0] trunc_ln10_reg_2080;
reg   [0:0] tmp_24_reg_2085;
wire   [0:0] tmp_24_reg_2085_pp0_iter11_reg;
reg   [0:0] tmp_25_reg_2089;
wire   [0:0] tmp_25_reg_2089_pp0_iter11_reg;
reg   [0:0] tmp_26_reg_2093;
wire   [0:0] tmp_26_reg_2093_pp0_iter11_reg;
reg   [0:0] tmp_27_reg_2097;
wire   [0:0] tmp_27_reg_2097_pp0_iter11_reg;
reg   [0:0] tmp_28_reg_2101;
wire   [0:0] tmp_28_reg_2101_pp0_iter11_reg;
reg   [0:0] tmp_29_reg_2105;
wire   [0:0] tmp_29_reg_2105_pp0_iter11_reg;
reg   [0:0] tmp_30_reg_2109;
wire   [0:0] tmp_30_reg_2109_pp0_iter11_reg;
reg   [0:0] tmp_31_reg_2113;
wire   [0:0] tmp_31_reg_2113_pp0_iter11_reg;
reg   [0:0] tmp_32_reg_2117;
wire   [0:0] tmp_32_reg_2117_pp0_iter11_reg;
reg   [0:0] tmp_33_reg_2121;
wire   [0:0] tmp_33_reg_2121_pp0_iter11_reg;
reg   [0:0] tmp_34_reg_2125;
wire   [0:0] tmp_34_reg_2125_pp0_iter11_reg;
reg   [0:0] tmp_35_reg_2129;
wire   [0:0] tmp_35_reg_2129_pp0_iter11_reg;
reg   [0:0] tmp_36_reg_2133;
wire   [0:0] tmp_36_reg_2133_pp0_iter11_reg;
reg   [0:0] tmp_37_reg_2137;
wire   [0:0] tmp_37_reg_2137_pp0_iter11_reg;
reg   [0:0] tmp_38_reg_2141;
wire   [0:0] tmp_38_reg_2141_pp0_iter11_reg;
reg   [0:0] tmp_39_reg_2145;
wire   [0:0] tmp_39_reg_2145_pp0_iter11_reg;
reg   [0:0] tmp_40_reg_2149;
wire   [0:0] tmp_40_reg_2149_pp0_iter11_reg;
reg   [0:0] tmp_41_reg_2153;
wire   [0:0] tmp_41_reg_2153_pp0_iter11_reg;
reg   [0:0] tmp_42_reg_2157;
wire   [0:0] tmp_42_reg_2157_pp0_iter11_reg;
reg   [0:0] tmp_43_reg_2161;
wire   [0:0] tmp_43_reg_2161_pp0_iter11_reg;
reg   [0:0] tmp_44_reg_2165;
wire   [0:0] tmp_44_reg_2165_pp0_iter11_reg;
reg   [0:0] tmp_45_reg_2169;
wire   [0:0] tmp_45_reg_2169_pp0_iter11_reg;
reg   [0:0] tmp_46_reg_2173;
wire   [0:0] tmp_46_reg_2173_pp0_iter11_reg;
reg   [0:0] tmp_47_reg_2177;
wire   [0:0] tmp_47_reg_2177_pp0_iter11_reg;
reg   [0:0] tmp_48_reg_2181;
wire   [0:0] tmp_48_reg_2181_pp0_iter11_reg;
reg   [0:0] tmp_49_reg_2185;
wire   [0:0] tmp_49_reg_2185_pp0_iter11_reg;
wire   [5:0] select_ln16_fu_1019_p3;
wire   [3:0] s_val_fu_1074_p3;
reg   [3:0] s_val_reg_2194;
reg   [3:0] s_val_reg_2194_pp0_iter14_reg;
reg   [3:0] s_val_reg_2194_pp0_iter15_reg;
reg   [3:0] s_val_reg_2194_pp0_iter16_reg;
wire  signed [4:0] sub_i_i_i_fu_1086_p2;
reg  signed [4:0] sub_i_i_i_reg_2200;
reg   [0:0] tmp_52_reg_2205;
reg   [0:0] tmp_52_reg_2205_pp0_iter14_reg;
wire   [4:0] conv_i_i9_i_i_fu_1100_p2;
reg   [4:0] conv_i_i9_i_i_reg_2217;
wire   [0:0] lnot_i_i_i_fu_1106_p2;
reg   [0:0] lnot_i_i_i_reg_2222;
wire   [0:0] lnot_i_i_i_reg_2222_pp0_iter14_reg;
wire   [26:0] shl_ln196_fu_1121_p2;
reg   [26:0] shl_ln196_reg_2226;
wire   [26:0] ashr_ln196_fu_1130_p2;
reg   [26:0] ashr_ln196_reg_2231;
wire   [26:0] shl_ln196_1_fu_1139_p2;
reg   [26:0] shl_ln196_1_reg_2236;
wire   [26:0] ashr_ln196_1_fu_1148_p2;
reg   [26:0] ashr_ln196_1_reg_2241;
wire   [26:0] shl_ln196_2_fu_1157_p2;
reg   [26:0] shl_ln196_2_reg_2246;
wire   [26:0] ashr_ln196_2_fu_1166_p2;
reg   [26:0] ashr_ln196_2_reg_2251;
wire   [26:0] shl_ln196_3_fu_1175_p2;
reg   [26:0] shl_ln196_3_reg_2256;
wire   [26:0] ashr_ln196_3_fu_1184_p2;
reg   [26:0] ashr_ln196_3_reg_2261;
wire   [26:0] shl_ln196_4_fu_1193_p2;
reg   [26:0] shl_ln196_4_reg_2266;
wire   [26:0] ashr_ln196_4_fu_1202_p2;
reg   [26:0] ashr_ln196_4_reg_2271;
wire   [26:0] shl_ln196_5_fu_1211_p2;
reg   [26:0] shl_ln196_5_reg_2276;
wire   [26:0] ashr_ln196_5_fu_1220_p2;
reg   [26:0] ashr_ln196_5_reg_2281;
wire   [26:0] shl_ln196_6_fu_1229_p2;
reg   [26:0] shl_ln196_6_reg_2286;
wire   [26:0] ashr_ln196_6_fu_1238_p2;
reg   [26:0] ashr_ln196_6_reg_2291;
wire   [26:0] shl_ln196_7_fu_1247_p2;
reg   [26:0] shl_ln196_7_reg_2296;
wire   [26:0] ashr_ln196_7_fu_1256_p2;
reg   [26:0] ashr_ln196_7_reg_2301;
wire  signed [26:0] sext_ln198_fu_1282_p1;
wire  signed [26:0] sext_ln198_1_fu_1307_p1;
wire  signed [26:0] sext_ln198_2_fu_1332_p1;
wire  signed [26:0] sext_ln198_3_fu_1357_p1;
wire  signed [26:0] sext_ln198_4_fu_1382_p1;
wire  signed [26:0] sext_ln198_5_fu_1407_p1;
wire  signed [26:0] sext_ln198_6_fu_1432_p1;
wire  signed [26:0] sext_ln198_7_fu_1457_p1;
wire   [7:0] select_ln201_1_fu_1501_p3;
reg   [7:0] select_ln201_1_reg_2346;
wire   [7:0] select_ln201_3_fu_1693_p3;
reg   [7:0] select_ln201_3_reg_2351;
wire   [7:0] select_ln201_5_fu_1741_p3;
reg   [7:0] select_ln201_5_reg_2356;
wire   [7:0] select_ln201_7_fu_1761_p3;
reg   [7:0] select_ln201_7_reg_2361;
wire   [7:0] select_ln201_9_fu_1781_p3;
reg   [7:0] select_ln201_9_reg_2366;
wire   [7:0] select_ln201_11_fu_1801_p3;
reg   [7:0] select_ln201_11_reg_2371;
wire   [7:0] select_ln201_13_fu_1817_p3;
reg   [7:0] select_ln201_13_reg_2376;
wire   [7:0] select_ln201_15_fu_1837_p3;
reg   [7:0] select_ln201_15_reg_2381;
wire   [5:0] ap_phi_reg_pp0_iter0_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter1_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter2_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter3_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter4_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter5_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter6_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter7_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter8_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter9_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter10_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter11_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter12_s_val_i_reg_260;
reg   [5:0] ap_phi_reg_pp0_iter13_s_val_i_reg_260;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0_6522_i_reg_371;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0_6522_i_reg_371;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0_4496498520_i_reg_380;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0_4496498520_i_reg_380;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0_2478480494500518_i_reg_389;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0_2478480494500518_i_reg_389;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0468470476482492502516_i_reg_398;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0468470476482492502516_i_reg_398;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0_1472474484490504514_i_reg_407;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0_1472474484490504514_i_reg_407;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0_3486488506512_i_reg_416;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0_3486488506512_i_reg_416;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_0_5508510_i_reg_425;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_0_5508510_i_reg_425;
wire   [26:0] ap_phi_reg_pp0_iter0_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter1_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter2_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter3_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter4_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter5_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter6_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter7_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter8_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter9_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter10_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter11_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter12_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter13_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter14_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter15_q_val_33_reg_434;
reg   [26:0] ap_phi_reg_pp0_iter16_q_val_33_reg_434;
reg   [18:0] vec_fu_230;
wire   [18:0] vec_4_fu_461_p2;
wire    ap_loop_init;
reg   [18:0] ap_sig_allocacmp_vec_3;
wire  signed [18:0] select_ln175_cast_fu_443_p1;
wire   [26:0] select_ln185_fu_615_p3;
wire   [0:0] icmp_ln224_fu_624_p2;
wire   [25:0] trunc_ln224_fu_620_p1;
wire   [26:0] sub_ln185_1_fu_638_p2;
wire   [26:0] zext_ln185_fu_649_p1;
wire   [0:0] icmp_ln224_1_fu_652_p2;
wire   [26:0] sub_ln185_2_fu_664_p2;
wire   [0:0] icmp_ln224_2_fu_675_p2;
wire   [26:0] sub_ln185_3_fu_685_p2;
wire   [0:0] icmp_ln224_3_fu_696_p2;
wire   [26:0] sub_ln185_4_fu_706_p2;
wire   [0:0] icmp_ln224_4_fu_717_p2;
wire   [26:0] sub_ln185_5_fu_727_p2;
wire   [0:0] icmp_ln224_5_fu_738_p2;
wire   [26:0] sub_ln185_6_fu_748_p2;
wire   [0:0] icmp_ln224_6_fu_759_p2;
wire   [26:0] sub_ln185_7_fu_769_p2;
wire   [0:0] icmp_ln224_7_fu_780_p2;
wire   [0:0] icmp_ln12_fu_790_p2;
wire   [26:0] x_15_fu_795_p2;
wire   [26:0] x_16_fu_800_p3;
wire   [1:0] tmp_51_fu_1034_p4;
wire   [0:0] tmp_50_fu_1026_p3;
wire   [0:0] xor_ln189_fu_1054_p2;
wire   [0:0] icmp_ln43_fu_1044_p2;
wire   [0:0] or_ln189_fu_1068_p2;
wire   [3:0] select_ln189_fu_1060_p3;
wire   [3:0] trunc_ln189_fu_1050_p1;
wire   [4:0] s_val_cast_i_fu_1082_p1;
wire  signed [31:0] conv_i_i9_i_cast_i_fu_1111_p1;
wire   [26:0] conv_i_i9_i_cast_icast_fu_1117_p1;
wire  signed [31:0] conv_i_i_i36_i_fu_1114_p1;
wire   [26:0] conv_i_i_i36_icast_fu_1126_p1;
wire   [26:0] conv_i_i9_i_cast_icast85_fu_1135_p1;
wire   [26:0] conv_i_i_i36_icast86_fu_1144_p1;
wire   [26:0] conv_i_i9_i_cast_icast87_fu_1153_p1;
wire   [26:0] conv_i_i_i36_icast88_fu_1162_p1;
wire   [26:0] conv_i_i9_i_cast_icast89_fu_1171_p1;
wire   [26:0] conv_i_i_i36_icast90_fu_1180_p1;
wire   [26:0] conv_i_i9_i_cast_icast91_fu_1189_p1;
wire   [26:0] conv_i_i_i36_icast92_fu_1198_p1;
wire   [26:0] conv_i_i9_i_cast_icast93_fu_1207_p1;
wire   [26:0] conv_i_i_i36_icast94_fu_1216_p1;
wire   [26:0] conv_i_i9_i_cast_icast95_fu_1225_p1;
wire   [26:0] conv_i_i_i36_icast96_fu_1234_p1;
wire   [26:0] conv_i_i9_i_cast_icast97_fu_1243_p1;
wire   [26:0] conv_i_i_i36_icast98_fu_1252_p1;
wire   [26:0] q_val_6_fu_1261_p3;
wire   [26:0] q_val_10_fu_1266_p2;
wire   [25:0] q_val_3_fu_1272_p4;
wire   [26:0] q_val_12_fu_1286_p3;
wire   [26:0] q_val_13_fu_1291_p2;
wire   [25:0] q_val_7_fu_1297_p4;
wire   [26:0] q_val_14_fu_1311_p3;
wire   [26:0] q_val_15_fu_1316_p2;
wire   [25:0] q_val_11_fu_1322_p4;
wire   [26:0] q_val_17_fu_1336_p3;
wire   [26:0] q_val_18_fu_1341_p2;
wire   [25:0] q_val_19_fu_1347_p4;
wire   [26:0] q_val_20_fu_1361_p3;
wire   [26:0] q_val_21_fu_1366_p2;
wire   [25:0] q_val_22_fu_1372_p4;
wire   [26:0] q_val_23_fu_1386_p3;
wire   [26:0] q_val_24_fu_1391_p2;
wire   [25:0] q_val_25_fu_1397_p4;
wire   [26:0] q_val_26_fu_1411_p3;
wire   [26:0] q_val_27_fu_1416_p2;
wire   [25:0] q_val_28_fu_1422_p4;
wire   [26:0] q_val_29_fu_1436_p3;
wire   [26:0] q_val_30_fu_1441_p2;
wire   [25:0] q_val_31_fu_1447_p4;
wire   [19:0] tmp_53_fu_1467_p4;
wire   [0:0] icmp_ln42_fu_1461_p2;
wire   [0:0] icmp_ln43_8_fu_1477_p2;
wire   [0:0] or_ln201_fu_1495_p2;
wire   [7:0] select_ln201_fu_1487_p3;
wire   [7:0] trunc_ln201_fu_1483_p1;
wire   [19:0] tmp_54_fu_1515_p4;
wire   [0:0] icmp_ln42_1_fu_1509_p2;
wire   [0:0] icmp_ln43_9_fu_1525_p2;
wire   [19:0] tmp_55_fu_1543_p4;
wire   [0:0] icmp_ln42_2_fu_1537_p2;
wire   [0:0] icmp_ln43_10_fu_1553_p2;
wire   [19:0] tmp_56_fu_1571_p4;
wire   [0:0] icmp_ln42_3_fu_1565_p2;
wire   [0:0] icmp_ln43_11_fu_1581_p2;
wire   [19:0] tmp_57_fu_1599_p4;
wire   [0:0] icmp_ln42_4_fu_1593_p2;
wire   [0:0] icmp_ln43_12_fu_1609_p2;
wire   [19:0] tmp_58_fu_1631_p4;
wire   [0:0] icmp_ln42_5_fu_1625_p2;
wire   [0:0] icmp_ln43_13_fu_1641_p2;
wire   [19:0] tmp_59_fu_1659_p4;
wire   [0:0] icmp_ln42_6_fu_1653_p2;
wire   [0:0] icmp_ln43_14_fu_1669_p2;
wire   [0:0] or_ln201_6_fu_1687_p2;
wire   [7:0] select_ln201_2_fu_1679_p3;
wire   [7:0] trunc_ln201_2_fu_1675_p1;
wire   [19:0] tmp_60_fu_1707_p4;
wire   [0:0] icmp_ln42_7_fu_1701_p2;
wire   [0:0] icmp_ln43_15_fu_1717_p2;
wire   [0:0] or_ln201_7_fu_1735_p2;
wire   [7:0] select_ln201_4_fu_1727_p3;
wire   [7:0] trunc_ln201_3_fu_1723_p1;
wire   [0:0] or_ln201_1_fu_1531_p2;
wire   [7:0] select_ln201_6_fu_1749_p3;
wire   [7:0] trunc_ln204_fu_1757_p1;
wire   [0:0] or_ln201_2_fu_1559_p2;
wire   [7:0] select_ln201_8_fu_1769_p3;
wire   [7:0] trunc_ln204_1_fu_1777_p1;
wire   [0:0] or_ln201_3_fu_1587_p2;
wire   [7:0] select_ln201_10_fu_1789_p3;
wire   [7:0] trunc_ln204_2_fu_1797_p1;
wire   [0:0] or_ln201_4_fu_1619_p2;
wire   [7:0] select_ln201_12_fu_1809_p3;
wire   [7:0] trunc_ln201_1_fu_1615_p1;
wire   [0:0] or_ln201_5_fu_1647_p2;
wire   [7:0] select_ln201_14_fu_1825_p3;
wire   [7:0] trunc_ln204_3_fu_1833_p1;
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
reg    ap_loop_exit_ready_pp0_iter13_reg;
reg    ap_loop_exit_ready_pp0_iter14_reg;
reg    ap_loop_exit_ready_pp0_iter15_reg;
reg    ap_loop_exit_ready_pp0_iter16_reg;
reg    ap_loop_exit_ready_pp0_iter17_reg;
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
reg   [1:0] ap_NS_iter13_fsm;
reg   [1:0] ap_NS_iter14_fsm;
reg   [1:0] ap_NS_iter15_fsm;
reg   [1:0] ap_NS_iter16_fsm;
reg   [1:0] ap_NS_iter17_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
wire    ap_ST_iter5_fsm_state6_blk;
wire    ap_ST_iter6_fsm_state7_blk;
wire    ap_ST_iter7_fsm_state8_blk;
wire    ap_ST_iter8_fsm_state9_blk;
wire    ap_ST_iter9_fsm_state10_blk;
wire    ap_ST_iter10_fsm_state11_blk;
wire    ap_ST_iter11_fsm_state12_blk;
wire    ap_ST_iter12_fsm_state13_blk;
wire    ap_ST_iter13_fsm_state14_blk;
wire    ap_ST_iter14_fsm_state15_blk;
wire    ap_ST_iter15_fsm_state16_blk;
wire    ap_ST_iter16_fsm_state17_blk;
reg    ap_ST_iter17_fsm_state18_blk;
wire    ap_start_int;
reg    ap_condition_1052;
reg    ap_condition_1060;
reg    ap_condition_1064;
reg    ap_condition_1068;
reg    ap_condition_1072;
reg    ap_condition_1076;
reg    ap_condition_1080;
reg    ap_condition_1084;
reg    ap_condition_1088;
reg    ap_condition_1092;
reg    ap_condition_1096;
reg    ap_condition_1100;
reg    ap_condition_1104;
reg    ap_condition_1108;
reg    ap_condition_1112;
reg    ap_condition_1116;
reg    ap_condition_1120;
reg    ap_condition_1124;
reg    ap_condition_1128;
reg    ap_condition_1132;
reg    ap_condition_1136;
reg    ap_condition_1140;
reg    ap_condition_1144;
reg    ap_condition_1148;
reg    ap_condition_1152;
reg    ap_condition_1156;
reg    ap_condition_345;
reg    ap_condition_349;
reg    ap_condition_351;
reg    ap_condition_283;
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
//#0 ap_CS_iter13_fsm = 2'd1;
//#0 ap_CS_iter14_fsm = 2'd1;
//#0 ap_CS_iter15_fsm = 2'd1;
//#0 ap_CS_iter16_fsm = 2'd1;
//#0 ap_CS_iter17_fsm = 2'd1;
//#0 vec_fu_230 = 19'd0;
//#0 ap_done_reg = 1'b0;
end

SILU_GELU_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_CS_iter13_fsm <= ap_ST_iter13_fsm_state0;
    end else begin
        ap_CS_iter13_fsm <= ap_NS_iter13_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter14_fsm <= ap_ST_iter14_fsm_state0;
    end else begin
        ap_CS_iter14_fsm <= ap_NS_iter14_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter15_fsm <= ap_ST_iter15_fsm_state0;
    end else begin
        ap_CS_iter15_fsm <= ap_NS_iter15_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter16_fsm <= ap_ST_iter16_fsm_state0;
    end else begin
        ap_CS_iter16_fsm <= ap_NS_iter16_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter17_fsm <= ap_ST_iter17_fsm_state0;
    end else begin
        ap_CS_iter17_fsm <= ap_NS_iter17_fsm;
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
        end else if ((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (ap_loop_exit_ready_pp0_iter17_reg == 1'b1) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (ap_loop_exit_ready_pp0_iter16_reg == 1'b0) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
        ap_loop_exit_ready_pp0_iter17_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter16_fsm_state17))) begin
        ap_loop_exit_ready_pp0_iter17_reg <= ap_loop_exit_ready_pp0_iter16_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_345)) begin
        if ((1'b1 == ap_condition_1156)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd59;
        end else if ((1'b1 == ap_condition_1152)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd60;
        end else if ((1'b1 == ap_condition_1148)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd61;
        end else if ((1'b1 == ap_condition_1144)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd62;
        end else if ((1'b1 == ap_condition_1140)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd63;
        end else if ((1'b1 == ap_condition_1136)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd0;
        end else if ((1'b1 == ap_condition_1132)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd1;
        end else if ((1'b1 == ap_condition_1128)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd2;
        end else if ((1'b1 == ap_condition_1124)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd3;
        end else if ((1'b1 == ap_condition_1120)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd4;
        end else if ((1'b1 == ap_condition_1116)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd5;
        end else if ((1'b1 == ap_condition_1112)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd6;
        end else if ((1'b1 == ap_condition_1108)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd7;
        end else if ((1'b1 == ap_condition_1104)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd8;
        end else if ((1'b1 == ap_condition_1100)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd9;
        end else if ((1'b1 == ap_condition_1096)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd10;
        end else if ((1'b1 == ap_condition_1092)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd11;
        end else if ((1'b1 == ap_condition_1088)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd12;
        end else if ((1'b1 == ap_condition_1084)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd13;
        end else if ((1'b1 == ap_condition_1080)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd14;
        end else if ((1'b1 == ap_condition_1076)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd15;
        end else if ((1'b1 == ap_condition_1072)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd16;
        end else if ((1'b1 == ap_condition_1068)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd17;
        end else if ((1'b1 == ap_condition_1064)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd18;
        end else if ((1'b1 == ap_condition_1060)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd19;
        end else if (((tmp_24_reg_2085_pp0_iter11_reg == 1'd1) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= 6'd20;
        end else if ((1'b1 == ap_condition_1052)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= select_ln16_fu_1019_p3;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter13_s_val_i_reg_260 <= ap_phi_reg_pp0_iter12_s_val_i_reg_260;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0468470476482492502516_i_reg_398 <= q_val_reg_1873_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter14_q_val_0468470476482492502516_i_reg_398;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0_1472474484490504514_i_reg_407 <= q_val_4_reg_1882_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter14_q_val_0_1472474484490504514_i_reg_407;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0_2478480494500518_i_reg_389 <= q_val_8_reg_1891_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter14_q_val_0_2478480494500518_i_reg_389;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0_3486488506512_i_reg_416 <= q_val_16_reg_1900_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter14_q_val_0_3486488506512_i_reg_416;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0_4496498520_i_reg_380 <= q_val_1_reg_1909_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter14_q_val_0_4496498520_i_reg_380;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0_5508510_i_reg_425 <= q_val_5_reg_1918_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter14_q_val_0_5508510_i_reg_425;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_0_6522_i_reg_371 <= q_val_9_reg_1927_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter14_q_val_0_6522_i_reg_371;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_349)) begin
        if (((lnot_i_i_i_fu_1106_p2 == 1'd1) & (icmp_ln175_reg_1869_pp0_iter13_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter15_q_val_33_reg_434 <= q_val_2_reg_1936_pp0_iter13_reg;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter15_q_val_33_reg_434 <= ap_phi_reg_pp0_iter14_q_val_33_reg_434;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0468470476482492502516_i_reg_398 <= sext_ln198_fu_1282_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter15_q_val_0468470476482492502516_i_reg_398;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0_1472474484490504514_i_reg_407 <= sext_ln198_1_fu_1307_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter15_q_val_0_1472474484490504514_i_reg_407;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0_2478480494500518_i_reg_389 <= sext_ln198_2_fu_1332_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter15_q_val_0_2478480494500518_i_reg_389;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0_3486488506512_i_reg_416 <= sext_ln198_3_fu_1357_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter15_q_val_0_3486488506512_i_reg_416;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0_4496498520_i_reg_380 <= sext_ln198_4_fu_1382_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter15_q_val_0_4496498520_i_reg_380;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0_5508510_i_reg_425 <= sext_ln198_5_fu_1407_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter15_q_val_0_5508510_i_reg_425;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_0_6522_i_reg_371 <= sext_ln198_6_fu_1432_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter15_q_val_0_6522_i_reg_371;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_351)) begin
        if (((lnot_i_i_i_reg_2222_pp0_iter14_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter14_reg == 1'd0))) begin
            ap_phi_reg_pp0_iter16_q_val_33_reg_434 <= sext_ln198_7_fu_1457_p1;
        end else if ((1'b1 == 1'b1)) begin
            ap_phi_reg_pp0_iter16_q_val_33_reg_434 <= ap_phi_reg_pp0_iter15_q_val_33_reg_434;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_283)) begin
        if ((icmp_ln175_fu_455_p2 == 1'd0)) begin
            vec_fu_230 <= vec_4_fu_461_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            vec_fu_230 <= 19'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter10_fsm_state11))) begin
        abs_max_reg_2073 <= abs_max_fu_784_p3;
        ap_loop_exit_ready_pp0_iter11_reg <= ap_loop_exit_ready_pp0_iter10_reg;
        ap_phi_reg_pp0_iter11_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter10_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter11_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter10_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter11_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter10_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter11_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter10_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter11_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter10_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter11_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter10_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter11_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter10_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter11_q_val_33_reg_434 <= ap_phi_reg_pp0_iter10_q_val_33_reg_434;
        ap_phi_reg_pp0_iter11_s_val_i_reg_260 <= ap_phi_reg_pp0_iter10_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter10_reg <= icmp_ln175_reg_1869_pp0_iter9_reg;
        q_val_16_reg_1900_pp0_iter10_reg <= q_val_16_reg_1900_pp0_iter9_reg;
        q_val_1_reg_1909_pp0_iter10_reg <= q_val_1_reg_1909_pp0_iter9_reg;
        q_val_2_reg_1936_pp0_iter10_reg <= q_val_2_reg_1936_pp0_iter9_reg;
        q_val_4_reg_1882_pp0_iter10_reg <= q_val_4_reg_1882_pp0_iter9_reg;
        q_val_5_reg_1918_pp0_iter10_reg <= q_val_5_reg_1918_pp0_iter9_reg;
        q_val_8_reg_1891_pp0_iter10_reg <= q_val_8_reg_1891_pp0_iter9_reg;
        q_val_9_reg_1927_pp0_iter10_reg <= q_val_9_reg_1927_pp0_iter9_reg;
        q_val_reg_1873_pp0_iter10_reg <= q_val_reg_1873_pp0_iter9_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter9_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter10_reg <= ap_loop_exit_ready_pp0_iter9_reg;
        ap_phi_reg_pp0_iter10_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter9_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter10_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter9_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter10_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter9_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter10_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter9_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter10_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter9_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter10_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter9_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter10_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter9_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter10_q_val_33_reg_434 <= ap_phi_reg_pp0_iter9_q_val_33_reg_434;
        ap_phi_reg_pp0_iter10_s_val_i_reg_260 <= ap_phi_reg_pp0_iter9_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter9_reg <= icmp_ln175_reg_1869_pp0_iter8_reg;
        q_val_16_reg_1900_pp0_iter9_reg <= q_val_16_reg_1900_pp0_iter8_reg;
        q_val_1_reg_1909_pp0_iter9_reg <= q_val_1_reg_1909_pp0_iter8_reg;
        q_val_2_reg_1936_pp0_iter9_reg <= q_val_2_reg_1936_pp0_iter8_reg;
        q_val_4_reg_1882_pp0_iter9_reg <= q_val_4_reg_1882_pp0_iter8_reg;
        q_val_5_reg_1918_pp0_iter9_reg <= q_val_5_reg_1918_pp0_iter8_reg;
        q_val_8_reg_1891_pp0_iter9_reg <= q_val_8_reg_1891_pp0_iter8_reg;
        q_val_9_reg_1927_pp0_iter9_reg <= q_val_9_reg_1927_pp0_iter8_reg;
        q_val_reg_1873_pp0_iter9_reg <= q_val_reg_1873_pp0_iter8_reg;
        select_ln185_13_reg_2061 <= select_ln185_13_fu_763_p3;
        select_ln185_14_reg_2067 <= select_ln185_14_fu_774_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
        ap_loop_exit_ready_pp0_iter12_reg <= ap_loop_exit_ready_pp0_iter11_reg;
        ap_phi_reg_pp0_iter12_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter11_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter12_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter11_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter12_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter11_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter12_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter11_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter12_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter11_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter12_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter11_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter12_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter11_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter12_q_val_33_reg_434 <= ap_phi_reg_pp0_iter11_q_val_33_reg_434;
        ap_phi_reg_pp0_iter12_s_val_i_reg_260 <= ap_phi_reg_pp0_iter11_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter11_reg <= icmp_ln175_reg_1869_pp0_iter10_reg;
        q_val_16_reg_1900_pp0_iter11_reg <= q_val_16_reg_1900_pp0_iter10_reg;
        q_val_1_reg_1909_pp0_iter11_reg <= q_val_1_reg_1909_pp0_iter10_reg;
        q_val_2_reg_1936_pp0_iter11_reg <= q_val_2_reg_1936_pp0_iter10_reg;
        q_val_4_reg_1882_pp0_iter11_reg <= q_val_4_reg_1882_pp0_iter10_reg;
        q_val_5_reg_1918_pp0_iter11_reg <= q_val_5_reg_1918_pp0_iter10_reg;
        q_val_8_reg_1891_pp0_iter11_reg <= q_val_8_reg_1891_pp0_iter10_reg;
        q_val_9_reg_1927_pp0_iter11_reg <= q_val_9_reg_1927_pp0_iter10_reg;
        q_val_reg_1873_pp0_iter11_reg <= q_val_reg_1873_pp0_iter10_reg;
        tmp_24_reg_2085 <= x_16_fu_800_p3[32'd26];
        tmp_25_reg_2089 <= x_16_fu_800_p3[32'd25];
        tmp_26_reg_2093 <= x_16_fu_800_p3[32'd24];
        tmp_27_reg_2097 <= x_16_fu_800_p3[32'd23];
        tmp_28_reg_2101 <= x_16_fu_800_p3[32'd22];
        tmp_29_reg_2105 <= x_16_fu_800_p3[32'd21];
        tmp_30_reg_2109 <= x_16_fu_800_p3[32'd20];
        tmp_31_reg_2113 <= x_16_fu_800_p3[32'd19];
        tmp_32_reg_2117 <= x_16_fu_800_p3[32'd18];
        tmp_33_reg_2121 <= x_16_fu_800_p3[32'd17];
        tmp_34_reg_2125 <= x_16_fu_800_p3[32'd16];
        tmp_35_reg_2129 <= x_16_fu_800_p3[32'd15];
        tmp_36_reg_2133 <= x_16_fu_800_p3[32'd14];
        tmp_37_reg_2137 <= x_16_fu_800_p3[32'd13];
        tmp_38_reg_2141 <= x_16_fu_800_p3[32'd12];
        tmp_39_reg_2145 <= x_16_fu_800_p3[32'd11];
        tmp_40_reg_2149 <= x_16_fu_800_p3[32'd10];
        tmp_41_reg_2153 <= x_16_fu_800_p3[32'd9];
        tmp_42_reg_2157 <= x_16_fu_800_p3[32'd8];
        tmp_43_reg_2161 <= x_16_fu_800_p3[32'd7];
        tmp_44_reg_2165 <= x_16_fu_800_p3[32'd6];
        tmp_45_reg_2169 <= x_16_fu_800_p3[32'd5];
        tmp_46_reg_2173 <= x_16_fu_800_p3[32'd4];
        tmp_47_reg_2177 <= x_16_fu_800_p3[32'd3];
        tmp_48_reg_2181 <= x_16_fu_800_p3[32'd2];
        tmp_49_reg_2185 <= x_16_fu_800_p3[32'd1];
        trunc_ln10_reg_2080 <= trunc_ln10_fu_807_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        ap_loop_exit_ready_pp0_iter13_reg <= ap_loop_exit_ready_pp0_iter12_reg;
        ap_phi_reg_pp0_iter13_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter12_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter13_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter12_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter13_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter12_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter13_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter12_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter13_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter12_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter13_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter12_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter13_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter12_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter13_q_val_33_reg_434 <= ap_phi_reg_pp0_iter12_q_val_33_reg_434;
        icmp_ln175_reg_1869_pp0_iter12_reg <= icmp_ln175_reg_1869_pp0_iter11_reg;
        q_val_16_reg_1900_pp0_iter12_reg <= q_val_16_reg_1900_pp0_iter11_reg;
        q_val_1_reg_1909_pp0_iter12_reg <= q_val_1_reg_1909_pp0_iter11_reg;
        q_val_2_reg_1936_pp0_iter12_reg <= q_val_2_reg_1936_pp0_iter11_reg;
        q_val_4_reg_1882_pp0_iter12_reg <= q_val_4_reg_1882_pp0_iter11_reg;
        q_val_5_reg_1918_pp0_iter12_reg <= q_val_5_reg_1918_pp0_iter11_reg;
        q_val_8_reg_1891_pp0_iter12_reg <= q_val_8_reg_1891_pp0_iter11_reg;
        q_val_9_reg_1927_pp0_iter12_reg <= q_val_9_reg_1927_pp0_iter11_reg;
        q_val_reg_1873_pp0_iter12_reg <= q_val_reg_1873_pp0_iter11_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter13_fsm_state14))) begin
        ap_loop_exit_ready_pp0_iter14_reg <= ap_loop_exit_ready_pp0_iter13_reg;
        ap_phi_reg_pp0_iter14_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter13_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter14_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter13_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter14_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter13_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter14_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter13_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter14_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter13_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter14_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter13_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter14_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter13_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter14_q_val_33_reg_434 <= ap_phi_reg_pp0_iter13_q_val_33_reg_434;
        conv_i_i9_i_i_reg_2217 <= conv_i_i9_i_i_fu_1100_p2;
        icmp_ln175_reg_1869_pp0_iter13_reg <= icmp_ln175_reg_1869_pp0_iter12_reg;
        q_val_16_reg_1900_pp0_iter13_reg <= q_val_16_reg_1900_pp0_iter12_reg;
        q_val_1_reg_1909_pp0_iter13_reg <= q_val_1_reg_1909_pp0_iter12_reg;
        q_val_2_reg_1936_pp0_iter13_reg <= q_val_2_reg_1936_pp0_iter12_reg;
        q_val_4_reg_1882_pp0_iter13_reg <= q_val_4_reg_1882_pp0_iter12_reg;
        q_val_5_reg_1918_pp0_iter13_reg <= q_val_5_reg_1918_pp0_iter12_reg;
        q_val_8_reg_1891_pp0_iter13_reg <= q_val_8_reg_1891_pp0_iter12_reg;
        q_val_9_reg_1927_pp0_iter13_reg <= q_val_9_reg_1927_pp0_iter12_reg;
        q_val_reg_1873_pp0_iter13_reg <= q_val_reg_1873_pp0_iter12_reg;
        s_val_reg_2194 <= s_val_fu_1074_p3;
        sub_i_i_i_reg_2200 <= sub_i_i_i_fu_1086_p2;
        tmp_52_reg_2205 <= sub_i_i_i_fu_1086_p2[32'd4];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter14_fsm_state15))) begin
        ap_loop_exit_ready_pp0_iter15_reg <= ap_loop_exit_ready_pp0_iter14_reg;
        ashr_ln196_1_reg_2241 <= ashr_ln196_1_fu_1148_p2;
        ashr_ln196_2_reg_2251 <= ashr_ln196_2_fu_1166_p2;
        ashr_ln196_3_reg_2261 <= ashr_ln196_3_fu_1184_p2;
        ashr_ln196_4_reg_2271 <= ashr_ln196_4_fu_1202_p2;
        ashr_ln196_5_reg_2281 <= ashr_ln196_5_fu_1220_p2;
        ashr_ln196_6_reg_2291 <= ashr_ln196_6_fu_1238_p2;
        ashr_ln196_7_reg_2301 <= ashr_ln196_7_fu_1256_p2;
        ashr_ln196_reg_2231 <= ashr_ln196_fu_1130_p2;
        icmp_ln175_reg_1869_pp0_iter14_reg <= icmp_ln175_reg_1869_pp0_iter13_reg;
        lnot_i_i_i_reg_2222 <= lnot_i_i_i_fu_1106_p2;
        s_val_reg_2194_pp0_iter14_reg <= s_val_reg_2194;
        shl_ln196_1_reg_2236 <= shl_ln196_1_fu_1139_p2;
        shl_ln196_2_reg_2246 <= shl_ln196_2_fu_1157_p2;
        shl_ln196_3_reg_2256 <= shl_ln196_3_fu_1175_p2;
        shl_ln196_4_reg_2266 <= shl_ln196_4_fu_1193_p2;
        shl_ln196_5_reg_2276 <= shl_ln196_5_fu_1211_p2;
        shl_ln196_6_reg_2286 <= shl_ln196_6_fu_1229_p2;
        shl_ln196_7_reg_2296 <= shl_ln196_7_fu_1247_p2;
        shl_ln196_reg_2226 <= shl_ln196_fu_1121_p2;
        tmp_52_reg_2205_pp0_iter14_reg <= tmp_52_reg_2205;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter15_fsm_state16))) begin
        ap_loop_exit_ready_pp0_iter16_reg <= ap_loop_exit_ready_pp0_iter15_reg;
        icmp_ln175_reg_1869_pp0_iter15_reg <= icmp_ln175_reg_1869_pp0_iter14_reg;
        s_val_reg_2194_pp0_iter15_reg <= s_val_reg_2194_pp0_iter14_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) | ((1'b1 == ap_block_state2_pp0_stage0_iter1) & (1'b1 == ap_CS_iter1_fsm_state2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        ap_phi_reg_pp0_iter1_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter0_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter1_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter0_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter1_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter0_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter1_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter0_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter1_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter0_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter1_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter0_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter1_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter0_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter1_q_val_33_reg_434 <= ap_phi_reg_pp0_iter0_q_val_33_reg_434;
        ap_phi_reg_pp0_iter1_s_val_i_reg_260 <= ap_phi_reg_pp0_iter0_s_val_i_reg_260;
        icmp_ln175_reg_1869 <= icmp_ln175_fu_455_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        ap_phi_reg_pp0_iter2_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter1_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter2_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter1_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter2_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter1_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter2_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter1_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter2_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter1_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter2_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter1_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter2_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter1_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter2_q_val_33_reg_434 <= ap_phi_reg_pp0_iter1_q_val_33_reg_434;
        ap_phi_reg_pp0_iter2_s_val_i_reg_260 <= ap_phi_reg_pp0_iter1_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter1_reg <= icmp_ln175_reg_1869;
        q_val_16_reg_1900 <= {{act_stream_i_dout[107:81]}};
        q_val_1_reg_1909 <= {{act_stream_i_dout[134:108]}};
        q_val_2_reg_1936 <= {{act_stream_i_dout[215:189]}};
        q_val_4_reg_1882 <= {{act_stream_i_dout[53:27]}};
        q_val_5_reg_1918 <= {{act_stream_i_dout[161:135]}};
        q_val_8_reg_1891 <= {{act_stream_i_dout[80:54]}};
        q_val_9_reg_1927 <= {{act_stream_i_dout[188:162]}};
        q_val_reg_1873 <= q_val_fu_472_p1;
        tmp_17_reg_1950 <= act_stream_i_dout[32'd53];
        tmp_18_reg_1955 <= act_stream_i_dout[32'd80];
        tmp_19_reg_1960 <= act_stream_i_dout[32'd107];
        tmp_20_reg_1965 <= act_stream_i_dout[32'd134];
        tmp_21_reg_1970 <= act_stream_i_dout[32'd161];
        tmp_22_reg_1975 <= act_stream_i_dout[32'd188];
        tmp_23_reg_1980 <= act_stream_i_dout[32'd215];
        tmp_reg_1945 <= act_stream_i_dout[32'd26];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        ap_phi_reg_pp0_iter3_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter2_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter3_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter2_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter3_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter2_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter3_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter2_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter3_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter2_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter3_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter2_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter3_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter2_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter3_q_val_33_reg_434 <= ap_phi_reg_pp0_iter2_q_val_33_reg_434;
        ap_phi_reg_pp0_iter3_s_val_i_reg_260 <= ap_phi_reg_pp0_iter2_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter2_reg <= icmp_ln175_reg_1869_pp0_iter1_reg;
        q_val_16_reg_1900_pp0_iter2_reg <= q_val_16_reg_1900;
        q_val_1_reg_1909_pp0_iter2_reg <= q_val_1_reg_1909;
        q_val_2_reg_1936_pp0_iter2_reg <= q_val_2_reg_1936;
        q_val_4_reg_1882_pp0_iter2_reg <= q_val_4_reg_1882;
        q_val_5_reg_1918_pp0_iter2_reg <= q_val_5_reg_1918;
        q_val_8_reg_1891_pp0_iter2_reg <= q_val_8_reg_1891;
        q_val_9_reg_1927_pp0_iter2_reg <= q_val_9_reg_1927;
        q_val_reg_1873_pp0_iter2_reg <= q_val_reg_1873;
        sub_ln185_reg_1985 <= sub_ln185_fu_610_p2;
        tmp_17_reg_1950_pp0_iter2_reg <= tmp_17_reg_1950;
        tmp_18_reg_1955_pp0_iter2_reg <= tmp_18_reg_1955;
        tmp_19_reg_1960_pp0_iter2_reg <= tmp_19_reg_1960;
        tmp_20_reg_1965_pp0_iter2_reg <= tmp_20_reg_1965;
        tmp_21_reg_1970_pp0_iter2_reg <= tmp_21_reg_1970;
        tmp_22_reg_1975_pp0_iter2_reg <= tmp_22_reg_1975;
        tmp_23_reg_1980_pp0_iter2_reg <= tmp_23_reg_1980;
        tmp_reg_1945_pp0_iter2_reg <= tmp_reg_1945;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        ap_phi_reg_pp0_iter4_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter3_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter4_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter3_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter4_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter3_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter4_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter3_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter4_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter3_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter4_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter3_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter4_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter3_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter4_q_val_33_reg_434 <= ap_phi_reg_pp0_iter3_q_val_33_reg_434;
        ap_phi_reg_pp0_iter4_s_val_i_reg_260 <= ap_phi_reg_pp0_iter3_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter3_reg <= icmp_ln175_reg_1869_pp0_iter2_reg;
        q_val_16_reg_1900_pp0_iter3_reg <= q_val_16_reg_1900_pp0_iter2_reg;
        q_val_1_reg_1909_pp0_iter3_reg <= q_val_1_reg_1909_pp0_iter2_reg;
        q_val_2_reg_1936_pp0_iter3_reg <= q_val_2_reg_1936_pp0_iter2_reg;
        q_val_4_reg_1882_pp0_iter3_reg <= q_val_4_reg_1882_pp0_iter2_reg;
        q_val_5_reg_1918_pp0_iter3_reg <= q_val_5_reg_1918_pp0_iter2_reg;
        q_val_8_reg_1891_pp0_iter3_reg <= q_val_8_reg_1891_pp0_iter2_reg;
        q_val_9_reg_1927_pp0_iter3_reg <= q_val_9_reg_1927_pp0_iter2_reg;
        q_val_reg_1873_pp0_iter3_reg <= q_val_reg_1873_pp0_iter2_reg;
        select_ln185_1_reg_1990 <= select_ln185_1_fu_630_p3;
        select_ln185_2_reg_1995 <= select_ln185_2_fu_643_p3;
        tmp_18_reg_1955_pp0_iter3_reg <= tmp_18_reg_1955_pp0_iter2_reg;
        tmp_19_reg_1960_pp0_iter3_reg <= tmp_19_reg_1960_pp0_iter2_reg;
        tmp_20_reg_1965_pp0_iter3_reg <= tmp_20_reg_1965_pp0_iter2_reg;
        tmp_21_reg_1970_pp0_iter3_reg <= tmp_21_reg_1970_pp0_iter2_reg;
        tmp_22_reg_1975_pp0_iter3_reg <= tmp_22_reg_1975_pp0_iter2_reg;
        tmp_23_reg_1980_pp0_iter3_reg <= tmp_23_reg_1980_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        ap_phi_reg_pp0_iter5_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter4_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter5_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter4_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter5_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter4_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter5_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter4_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter5_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter4_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter5_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter4_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter5_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter4_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter5_q_val_33_reg_434 <= ap_phi_reg_pp0_iter4_q_val_33_reg_434;
        ap_phi_reg_pp0_iter5_s_val_i_reg_260 <= ap_phi_reg_pp0_iter4_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter4_reg <= icmp_ln175_reg_1869_pp0_iter3_reg;
        q_val_16_reg_1900_pp0_iter4_reg <= q_val_16_reg_1900_pp0_iter3_reg;
        q_val_1_reg_1909_pp0_iter4_reg <= q_val_1_reg_1909_pp0_iter3_reg;
        q_val_2_reg_1936_pp0_iter4_reg <= q_val_2_reg_1936_pp0_iter3_reg;
        q_val_4_reg_1882_pp0_iter4_reg <= q_val_4_reg_1882_pp0_iter3_reg;
        q_val_5_reg_1918_pp0_iter4_reg <= q_val_5_reg_1918_pp0_iter3_reg;
        q_val_8_reg_1891_pp0_iter4_reg <= q_val_8_reg_1891_pp0_iter3_reg;
        q_val_9_reg_1927_pp0_iter4_reg <= q_val_9_reg_1927_pp0_iter3_reg;
        q_val_reg_1873_pp0_iter4_reg <= q_val_reg_1873_pp0_iter3_reg;
        select_ln185_3_reg_2001 <= select_ln185_3_fu_657_p3;
        select_ln185_4_reg_2007 <= select_ln185_4_fu_669_p3;
        tmp_19_reg_1960_pp0_iter4_reg <= tmp_19_reg_1960_pp0_iter3_reg;
        tmp_20_reg_1965_pp0_iter4_reg <= tmp_20_reg_1965_pp0_iter3_reg;
        tmp_21_reg_1970_pp0_iter4_reg <= tmp_21_reg_1970_pp0_iter3_reg;
        tmp_22_reg_1975_pp0_iter4_reg <= tmp_22_reg_1975_pp0_iter3_reg;
        tmp_23_reg_1980_pp0_iter4_reg <= tmp_23_reg_1980_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        ap_phi_reg_pp0_iter6_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter5_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter6_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter5_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter6_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter5_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter6_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter5_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter6_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter5_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter6_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter5_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter6_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter5_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter6_q_val_33_reg_434 <= ap_phi_reg_pp0_iter5_q_val_33_reg_434;
        ap_phi_reg_pp0_iter6_s_val_i_reg_260 <= ap_phi_reg_pp0_iter5_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter5_reg <= icmp_ln175_reg_1869_pp0_iter4_reg;
        q_val_16_reg_1900_pp0_iter5_reg <= q_val_16_reg_1900_pp0_iter4_reg;
        q_val_1_reg_1909_pp0_iter5_reg <= q_val_1_reg_1909_pp0_iter4_reg;
        q_val_2_reg_1936_pp0_iter5_reg <= q_val_2_reg_1936_pp0_iter4_reg;
        q_val_4_reg_1882_pp0_iter5_reg <= q_val_4_reg_1882_pp0_iter4_reg;
        q_val_5_reg_1918_pp0_iter5_reg <= q_val_5_reg_1918_pp0_iter4_reg;
        q_val_8_reg_1891_pp0_iter5_reg <= q_val_8_reg_1891_pp0_iter4_reg;
        q_val_9_reg_1927_pp0_iter5_reg <= q_val_9_reg_1927_pp0_iter4_reg;
        q_val_reg_1873_pp0_iter5_reg <= q_val_reg_1873_pp0_iter4_reg;
        select_ln185_5_reg_2013 <= select_ln185_5_fu_679_p3;
        select_ln185_6_reg_2019 <= select_ln185_6_fu_690_p3;
        tmp_20_reg_1965_pp0_iter5_reg <= tmp_20_reg_1965_pp0_iter4_reg;
        tmp_21_reg_1970_pp0_iter5_reg <= tmp_21_reg_1970_pp0_iter4_reg;
        tmp_22_reg_1975_pp0_iter5_reg <= tmp_22_reg_1975_pp0_iter4_reg;
        tmp_23_reg_1980_pp0_iter5_reg <= tmp_23_reg_1980_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
        ap_phi_reg_pp0_iter7_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter6_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter7_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter6_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter7_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter6_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter7_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter6_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter7_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter6_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter7_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter6_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter7_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter6_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter7_q_val_33_reg_434 <= ap_phi_reg_pp0_iter6_q_val_33_reg_434;
        ap_phi_reg_pp0_iter7_s_val_i_reg_260 <= ap_phi_reg_pp0_iter6_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter6_reg <= icmp_ln175_reg_1869_pp0_iter5_reg;
        q_val_16_reg_1900_pp0_iter6_reg <= q_val_16_reg_1900_pp0_iter5_reg;
        q_val_1_reg_1909_pp0_iter6_reg <= q_val_1_reg_1909_pp0_iter5_reg;
        q_val_2_reg_1936_pp0_iter6_reg <= q_val_2_reg_1936_pp0_iter5_reg;
        q_val_4_reg_1882_pp0_iter6_reg <= q_val_4_reg_1882_pp0_iter5_reg;
        q_val_5_reg_1918_pp0_iter6_reg <= q_val_5_reg_1918_pp0_iter5_reg;
        q_val_8_reg_1891_pp0_iter6_reg <= q_val_8_reg_1891_pp0_iter5_reg;
        q_val_9_reg_1927_pp0_iter6_reg <= q_val_9_reg_1927_pp0_iter5_reg;
        q_val_reg_1873_pp0_iter6_reg <= q_val_reg_1873_pp0_iter5_reg;
        select_ln185_7_reg_2025 <= select_ln185_7_fu_700_p3;
        select_ln185_8_reg_2031 <= select_ln185_8_fu_711_p3;
        tmp_21_reg_1970_pp0_iter6_reg <= tmp_21_reg_1970_pp0_iter5_reg;
        tmp_22_reg_1975_pp0_iter6_reg <= tmp_22_reg_1975_pp0_iter5_reg;
        tmp_23_reg_1980_pp0_iter6_reg <= tmp_23_reg_1980_pp0_iter5_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter7_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter8_reg <= ap_loop_exit_ready_pp0_iter7_reg;
        ap_phi_reg_pp0_iter8_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter7_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter8_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter7_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter8_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter7_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter8_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter7_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter8_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter7_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter8_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter7_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter8_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter7_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter8_q_val_33_reg_434 <= ap_phi_reg_pp0_iter7_q_val_33_reg_434;
        ap_phi_reg_pp0_iter8_s_val_i_reg_260 <= ap_phi_reg_pp0_iter7_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter7_reg <= icmp_ln175_reg_1869_pp0_iter6_reg;
        q_val_16_reg_1900_pp0_iter7_reg <= q_val_16_reg_1900_pp0_iter6_reg;
        q_val_1_reg_1909_pp0_iter7_reg <= q_val_1_reg_1909_pp0_iter6_reg;
        q_val_2_reg_1936_pp0_iter7_reg <= q_val_2_reg_1936_pp0_iter6_reg;
        q_val_4_reg_1882_pp0_iter7_reg <= q_val_4_reg_1882_pp0_iter6_reg;
        q_val_5_reg_1918_pp0_iter7_reg <= q_val_5_reg_1918_pp0_iter6_reg;
        q_val_8_reg_1891_pp0_iter7_reg <= q_val_8_reg_1891_pp0_iter6_reg;
        q_val_9_reg_1927_pp0_iter7_reg <= q_val_9_reg_1927_pp0_iter6_reg;
        q_val_reg_1873_pp0_iter7_reg <= q_val_reg_1873_pp0_iter6_reg;
        select_ln185_10_reg_2043 <= select_ln185_10_fu_732_p3;
        select_ln185_9_reg_2037 <= select_ln185_9_fu_721_p3;
        tmp_22_reg_1975_pp0_iter7_reg <= tmp_22_reg_1975_pp0_iter6_reg;
        tmp_23_reg_1980_pp0_iter7_reg <= tmp_23_reg_1980_pp0_iter6_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter8_fsm_state9))) begin
        ap_loop_exit_ready_pp0_iter9_reg <= ap_loop_exit_ready_pp0_iter8_reg;
        ap_phi_reg_pp0_iter9_q_val_0468470476482492502516_i_reg_398 <= ap_phi_reg_pp0_iter8_q_val_0468470476482492502516_i_reg_398;
        ap_phi_reg_pp0_iter9_q_val_0_1472474484490504514_i_reg_407 <= ap_phi_reg_pp0_iter8_q_val_0_1472474484490504514_i_reg_407;
        ap_phi_reg_pp0_iter9_q_val_0_2478480494500518_i_reg_389 <= ap_phi_reg_pp0_iter8_q_val_0_2478480494500518_i_reg_389;
        ap_phi_reg_pp0_iter9_q_val_0_3486488506512_i_reg_416 <= ap_phi_reg_pp0_iter8_q_val_0_3486488506512_i_reg_416;
        ap_phi_reg_pp0_iter9_q_val_0_4496498520_i_reg_380 <= ap_phi_reg_pp0_iter8_q_val_0_4496498520_i_reg_380;
        ap_phi_reg_pp0_iter9_q_val_0_5508510_i_reg_425 <= ap_phi_reg_pp0_iter8_q_val_0_5508510_i_reg_425;
        ap_phi_reg_pp0_iter9_q_val_0_6522_i_reg_371 <= ap_phi_reg_pp0_iter8_q_val_0_6522_i_reg_371;
        ap_phi_reg_pp0_iter9_q_val_33_reg_434 <= ap_phi_reg_pp0_iter8_q_val_33_reg_434;
        ap_phi_reg_pp0_iter9_s_val_i_reg_260 <= ap_phi_reg_pp0_iter8_s_val_i_reg_260;
        icmp_ln175_reg_1869_pp0_iter8_reg <= icmp_ln175_reg_1869_pp0_iter7_reg;
        q_val_16_reg_1900_pp0_iter8_reg <= q_val_16_reg_1900_pp0_iter7_reg;
        q_val_1_reg_1909_pp0_iter8_reg <= q_val_1_reg_1909_pp0_iter7_reg;
        q_val_2_reg_1936_pp0_iter8_reg <= q_val_2_reg_1936_pp0_iter7_reg;
        q_val_4_reg_1882_pp0_iter8_reg <= q_val_4_reg_1882_pp0_iter7_reg;
        q_val_5_reg_1918_pp0_iter8_reg <= q_val_5_reg_1918_pp0_iter7_reg;
        q_val_8_reg_1891_pp0_iter8_reg <= q_val_8_reg_1891_pp0_iter7_reg;
        q_val_9_reg_1927_pp0_iter8_reg <= q_val_9_reg_1927_pp0_iter7_reg;
        q_val_reg_1873_pp0_iter8_reg <= q_val_reg_1873_pp0_iter7_reg;
        select_ln185_11_reg_2049 <= select_ln185_11_fu_742_p3;
        select_ln185_12_reg_2055 <= select_ln185_12_fu_753_p3;
        tmp_23_reg_1980_pp0_iter8_reg <= tmp_23_reg_1980_pp0_iter7_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter16_fsm_state17))) begin
        icmp_ln175_reg_1869_pp0_iter16_reg <= icmp_ln175_reg_1869_pp0_iter15_reg;
        s_val_reg_2194_pp0_iter16_reg <= s_val_reg_2194_pp0_iter15_reg;
        select_ln201_11_reg_2371 <= select_ln201_11_fu_1801_p3;
        select_ln201_13_reg_2376 <= select_ln201_13_fu_1817_p3;
        select_ln201_15_reg_2381 <= select_ln201_15_fu_1837_p3;
        select_ln201_1_reg_2346 <= select_ln201_1_fu_1501_p3;
        select_ln201_3_reg_2351 <= select_ln201_3_fu_1693_p3;
        select_ln201_5_reg_2356 <= select_ln201_5_fu_1741_p3;
        select_ln201_7_reg_2361 <= select_ln201_7_fu_1761_p3;
        select_ln201_9_reg_2366 <= select_ln201_9_fu_1781_p3;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln175_reg_1869 == 1'd0))) begin
        act_stream_i_blk_n = act_stream_i_empty_n;
    end else begin
        act_stream_i_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)))) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln175_reg_1869 == 1'd0))) begin
        act_stream_i_read = 1'b1;
    end else begin
        act_stream_i_read = 1'b0;
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

assign ap_ST_iter11_fsm_state12_blk = 1'b0;

assign ap_ST_iter12_fsm_state13_blk = 1'b0;

assign ap_ST_iter13_fsm_state14_blk = 1'b0;

assign ap_ST_iter14_fsm_state15_blk = 1'b0;

assign ap_ST_iter15_fsm_state16_blk = 1'b0;

assign ap_ST_iter16_fsm_state17_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) begin
        ap_ST_iter17_fsm_state18_blk = 1'b1;
    end else begin
        ap_ST_iter17_fsm_state18_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state2_pp0_stage0_iter1)) begin
        ap_ST_iter1_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_iter2_fsm_state3_blk = 1'b0;

assign ap_ST_iter3_fsm_state4_blk = 1'b0;

assign ap_ST_iter4_fsm_state5_blk = 1'b0;

assign ap_ST_iter5_fsm_state6_blk = 1'b0;

assign ap_ST_iter6_fsm_state7_blk = 1'b0;

assign ap_ST_iter7_fsm_state8_blk = 1'b0;

assign ap_ST_iter8_fsm_state9_blk = 1'b0;

assign ap_ST_iter9_fsm_state10_blk = 1'b0;

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) | ((1'b1 == ap_block_state2_pp0_stage0_iter1) & (1'b1 == ap_CS_iter1_fsm_state2))) & (icmp_ln175_fu_455_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (ap_loop_exit_ready_pp0_iter17_reg == 1'b1) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_CS_iter17_fsm_state0) & (1'b1 == ap_CS_iter16_fsm_state0) & (1'b1 == ap_CS_iter15_fsm_state0) & (1'b1 == ap_CS_iter14_fsm_state0) & (1'b1 == ap_CS_iter13_fsm_state0) & (1'b1 == ap_CS_iter12_fsm_state0) & (1'b1 == ap_CS_iter11_fsm_state0) & (1'b1 == ap_CS_iter10_fsm_state0) & (1'b1 == ap_CS_iter9_fsm_state0) & (1'b1 == ap_CS_iter8_fsm_state0) & (1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) | ((1'b1 == ap_block_state2_pp0_stage0_iter1) & (1'b1 == ap_CS_iter1_fsm_state2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_vec_3 = 19'd0;
    end else begin
        ap_sig_allocacmp_vec_3 = vec_fu_230;
    end
end

always @ (*) begin
    if (((icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
        q_stream_TDATA_blk_n = q_stream_TREADY;
    end else begin
        q_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
        q_stream_TVALID = 1'b1;
    end else begin
        q_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
        s_stream_TDATA_blk_n = s_stream_TREADY;
    end else begin
        s_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0) & (1'b1 == ap_CS_iter17_fsm_state18))) begin
        s_stream_TVALID = 1'b1;
    end else begin
        s_stream_TVALID = 1'b0;
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
            if ((~((1'b1 == ap_block_state2_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_block_state2_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) | ((1'b1 == ap_block_state2_pp0_stage0_iter1) & (1'b1 == ap_CS_iter1_fsm_state2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state2_pp0_stage0_iter1))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & ((1'b0 == ap_CS_iter1_fsm_state2) | ((1'b1 == ap_block_state2_pp0_stage0_iter1) & (1'b1 == ap_CS_iter1_fsm_state2))))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state2_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter6_fsm_state7))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter7_fsm_state8))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter7_fsm_state8))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state0;
            end else begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end
        end
        ap_ST_iter8_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter7_fsm_state8))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter8_fsm_state9))) begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state10;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter8_fsm_state9))) begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state0;
            end else begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state10;
            end
        end
        ap_ST_iter9_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter8_fsm_state9))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter9_fsm_state10))) begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state11;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter9_fsm_state10))) begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state0;
            end else begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state11;
            end
        end
        ap_ST_iter10_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter9_fsm_state10))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter10_fsm_state11))) begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state12;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter10_fsm_state11))) begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state0;
            end else begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state12;
            end
        end
        ap_ST_iter11_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter10_fsm_state11))) begin
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
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state13;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter11_fsm_state12))) begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state0;
            end else begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state13;
            end
        end
        ap_ST_iter12_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
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

always @ (*) begin
    case (ap_CS_iter13_fsm)
        ap_ST_iter13_fsm_state14 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
                ap_NS_iter13_fsm = ap_ST_iter13_fsm_state14;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter12_fsm_state13))) begin
                ap_NS_iter13_fsm = ap_ST_iter13_fsm_state0;
            end else begin
                ap_NS_iter13_fsm = ap_ST_iter13_fsm_state14;
            end
        end
        ap_ST_iter13_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
                ap_NS_iter13_fsm = ap_ST_iter13_fsm_state14;
            end else begin
                ap_NS_iter13_fsm = ap_ST_iter13_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter13_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter14_fsm)
        ap_ST_iter14_fsm_state15 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter13_fsm_state14))) begin
                ap_NS_iter14_fsm = ap_ST_iter14_fsm_state15;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter13_fsm_state14))) begin
                ap_NS_iter14_fsm = ap_ST_iter14_fsm_state0;
            end else begin
                ap_NS_iter14_fsm = ap_ST_iter14_fsm_state15;
            end
        end
        ap_ST_iter14_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter13_fsm_state14))) begin
                ap_NS_iter14_fsm = ap_ST_iter14_fsm_state15;
            end else begin
                ap_NS_iter14_fsm = ap_ST_iter14_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter14_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter15_fsm)
        ap_ST_iter15_fsm_state16 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter14_fsm_state15))) begin
                ap_NS_iter15_fsm = ap_ST_iter15_fsm_state16;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter14_fsm_state15))) begin
                ap_NS_iter15_fsm = ap_ST_iter15_fsm_state0;
            end else begin
                ap_NS_iter15_fsm = ap_ST_iter15_fsm_state16;
            end
        end
        ap_ST_iter15_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter14_fsm_state15))) begin
                ap_NS_iter15_fsm = ap_ST_iter15_fsm_state16;
            end else begin
                ap_NS_iter15_fsm = ap_ST_iter15_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter15_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter16_fsm)
        ap_ST_iter16_fsm_state17 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter15_fsm_state16))) begin
                ap_NS_iter16_fsm = ap_ST_iter16_fsm_state17;
            end else if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b0 == ap_CS_iter15_fsm_state16))) begin
                ap_NS_iter16_fsm = ap_ST_iter16_fsm_state0;
            end else begin
                ap_NS_iter16_fsm = ap_ST_iter16_fsm_state17;
            end
        end
        ap_ST_iter16_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter15_fsm_state16))) begin
                ap_NS_iter16_fsm = ap_ST_iter16_fsm_state17;
            end else begin
                ap_NS_iter16_fsm = ap_ST_iter16_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter16_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter17_fsm)
        ap_ST_iter17_fsm_state18 : begin
            if ((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (1'b0 == ap_CS_iter16_fsm_state17))) begin
                ap_NS_iter17_fsm = ap_ST_iter17_fsm_state0;
            end else if (((~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (1'b1 == ap_CS_iter16_fsm_state17)) | (~((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17)) & (icmp_ln175_reg_1869_pp0_iter16_reg == 1'd1) & (1'b1 == ap_CS_iter17_fsm_state18)))) begin
                ap_NS_iter17_fsm = ap_ST_iter17_fsm_state18;
            end else begin
                ap_NS_iter17_fsm = ap_ST_iter17_fsm_state18;
            end
        end
        ap_ST_iter17_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter16_fsm_state17))) begin
                ap_NS_iter17_fsm = ap_ST_iter17_fsm_state18;
            end else begin
                ap_NS_iter17_fsm = ap_ST_iter17_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter17_fsm = 'bx;
        end
    endcase
end

assign abs_max_fu_784_p3 = ((icmp_ln224_7_fu_780_p2[0:0] == 1'b1) ? select_ln185_14_reg_2067 : select_ln185_13_reg_2061);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter10_fsm_state0 = ap_CS_iter10_fsm[32'd0];

assign ap_CS_iter10_fsm_state11 = ap_CS_iter10_fsm[32'd1];

assign ap_CS_iter11_fsm_state0 = ap_CS_iter11_fsm[32'd0];

assign ap_CS_iter11_fsm_state12 = ap_CS_iter11_fsm[32'd1];

assign ap_CS_iter12_fsm_state0 = ap_CS_iter12_fsm[32'd0];

assign ap_CS_iter12_fsm_state13 = ap_CS_iter12_fsm[32'd1];

assign ap_CS_iter13_fsm_state0 = ap_CS_iter13_fsm[32'd0];

assign ap_CS_iter13_fsm_state14 = ap_CS_iter13_fsm[32'd1];

assign ap_CS_iter14_fsm_state0 = ap_CS_iter14_fsm[32'd0];

assign ap_CS_iter14_fsm_state15 = ap_CS_iter14_fsm[32'd1];

assign ap_CS_iter15_fsm_state0 = ap_CS_iter15_fsm[32'd0];

assign ap_CS_iter15_fsm_state16 = ap_CS_iter15_fsm[32'd1];

assign ap_CS_iter16_fsm_state0 = ap_CS_iter16_fsm[32'd0];

assign ap_CS_iter16_fsm_state17 = ap_CS_iter16_fsm[32'd1];

assign ap_CS_iter17_fsm_state0 = ap_CS_iter17_fsm[32'd0];

assign ap_CS_iter17_fsm_state18 = ap_CS_iter17_fsm[32'd1];

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
    ap_block_state18_io = (((s_stream_TREADY == 1'b0) & (icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0)) | ((icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0) & (q_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state18_pp0_stage0_iter17 = (((s_stream_TREADY == 1'b0) & (icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0)) | ((icmp_ln175_reg_1869_pp0_iter16_reg == 1'd0) & (q_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((1'b0 == act_stream_i_empty_n) & (icmp_ln175_reg_1869 == 1'd0));
end

always @ (*) begin
    ap_condition_1052 = ((tmp_49_reg_2185_pp0_iter11_reg == 1'd0) & (tmp_48_reg_2181_pp0_iter11_reg == 1'd0) & (tmp_47_reg_2177_pp0_iter11_reg == 1'd0) & (tmp_46_reg_2173_pp0_iter11_reg == 1'd0) & (tmp_45_reg_2169_pp0_iter11_reg == 1'd0) & (tmp_44_reg_2165_pp0_iter11_reg == 1'd0) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 
    1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1060 = ((tmp_25_reg_2089_pp0_iter11_reg == 1'd1) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1064 = ((tmp_26_reg_2093_pp0_iter11_reg == 1'd1) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1068 = ((tmp_27_reg_2097_pp0_iter11_reg == 1'd1) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1072 = ((tmp_28_reg_2101_pp0_iter11_reg == 1'd1) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1076 = ((tmp_29_reg_2105_pp0_iter11_reg == 1'd1) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1080 = ((tmp_30_reg_2109_pp0_iter11_reg == 1'd1) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1084 = ((tmp_31_reg_2113_pp0_iter11_reg == 1'd1) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1088 = ((tmp_32_reg_2117_pp0_iter11_reg == 1'd1) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1092 = ((tmp_33_reg_2121_pp0_iter11_reg == 1'd1) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1096 = ((tmp_34_reg_2125_pp0_iter11_reg == 1'd1) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1100 = ((tmp_35_reg_2129_pp0_iter11_reg == 1'd1) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1104 = ((tmp_36_reg_2133_pp0_iter11_reg == 1'd1) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1108 = ((tmp_37_reg_2137_pp0_iter11_reg == 1'd1) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1112 = ((tmp_38_reg_2141_pp0_iter11_reg == 1'd1) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1116 = ((tmp_39_reg_2145_pp0_iter11_reg == 1'd1) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1120 = ((tmp_40_reg_2149_pp0_iter11_reg == 1'd1) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1124 = ((tmp_41_reg_2153_pp0_iter11_reg == 1'd1) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1128 = ((tmp_42_reg_2157_pp0_iter11_reg == 1'd1) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1132 = ((tmp_43_reg_2161_pp0_iter11_reg == 1'd1) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1136 = ((tmp_44_reg_2165_pp0_iter11_reg == 1'd1) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1140 = ((tmp_45_reg_2169_pp0_iter11_reg == 1'd1) & (tmp_44_reg_2165_pp0_iter11_reg == 1'd0) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1144 = ((tmp_46_reg_2173_pp0_iter11_reg == 1'd1) & (tmp_45_reg_2169_pp0_iter11_reg == 1'd0) & (tmp_44_reg_2165_pp0_iter11_reg == 1'd0) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg 
    == 1'd0));
end

always @ (*) begin
    ap_condition_1148 = ((tmp_47_reg_2177_pp0_iter11_reg == 1'd1) & (tmp_46_reg_2173_pp0_iter11_reg == 1'd0) & (tmp_45_reg_2169_pp0_iter11_reg == 1'd0) & (tmp_44_reg_2165_pp0_iter11_reg == 1'd0) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 
    1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1152 = ((tmp_48_reg_2181_pp0_iter11_reg == 1'd1) & (tmp_47_reg_2177_pp0_iter11_reg == 1'd0) & (tmp_46_reg_2173_pp0_iter11_reg == 1'd0) & (tmp_45_reg_2169_pp0_iter11_reg == 1'd0) & (tmp_44_reg_2165_pp0_iter11_reg == 1'd0) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 
    1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_1156 = ((tmp_49_reg_2185_pp0_iter11_reg == 1'd1) & (tmp_48_reg_2181_pp0_iter11_reg == 1'd0) & (tmp_47_reg_2177_pp0_iter11_reg == 1'd0) & (tmp_46_reg_2173_pp0_iter11_reg == 1'd0) & (tmp_45_reg_2169_pp0_iter11_reg == 1'd0) & (tmp_44_reg_2165_pp0_iter11_reg == 1'd0) & (tmp_43_reg_2161_pp0_iter11_reg == 1'd0) & (tmp_42_reg_2157_pp0_iter11_reg == 1'd0) & (tmp_41_reg_2153_pp0_iter11_reg == 1'd0) & (tmp_40_reg_2149_pp0_iter11_reg == 1'd0) & (tmp_39_reg_2145_pp0_iter11_reg == 1'd0) & (tmp_38_reg_2141_pp0_iter11_reg == 1'd0) & (tmp_37_reg_2137_pp0_iter11_reg == 1'd0) & (tmp_36_reg_2133_pp0_iter11_reg == 1'd0) & (tmp_35_reg_2129_pp0_iter11_reg == 1'd0) & (tmp_34_reg_2125_pp0_iter11_reg == 1'd0) & (tmp_33_reg_2121_pp0_iter11_reg == 1'd0) & (tmp_32_reg_2117_pp0_iter11_reg == 1'd0) & (tmp_31_reg_2113_pp0_iter11_reg == 1'd0) & (tmp_30_reg_2109_pp0_iter11_reg == 1'd0) & (tmp_29_reg_2105_pp0_iter11_reg == 1'd0) & (tmp_28_reg_2101_pp0_iter11_reg == 1'd0) & (tmp_27_reg_2097_pp0_iter11_reg == 1'd0) & (tmp_26_reg_2093_pp0_iter11_reg == 
    1'd0) & (tmp_25_reg_2089_pp0_iter11_reg == 1'd0) & (tmp_24_reg_2085_pp0_iter11_reg == 1'd0) & (icmp_ln175_reg_1869_pp0_iter11_reg == 1'd0));
end

always @ (*) begin
    ap_condition_283 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) | ((1'b1 == ap_block_state2_pp0_stage0_iter1) & (1'b1 == ap_CS_iter1_fsm_state2))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_345 = (~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter12_fsm_state13));
end

always @ (*) begin
    ap_condition_349 = (~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter14_fsm_state15));
end

always @ (*) begin
    ap_condition_351 = (~((1'b1 == ap_CS_iter17_fsm_state18) & ((1'b1 == ap_block_state18_io) | (1'b1 == ap_block_state18_pp0_stage0_iter17))) & (1'b1 == ap_CS_iter15_fsm_state16));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_phi_reg_pp0_iter0_q_val_0468470476482492502516_i_reg_398 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_1472474484490504514_i_reg_407 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_2478480494500518_i_reg_389 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_3486488506512_i_reg_416 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_4496498520_i_reg_380 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_5508510_i_reg_425 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_0_6522_i_reg_371 = 'bx;

assign ap_phi_reg_pp0_iter0_q_val_33_reg_434 = 'bx;

assign ap_phi_reg_pp0_iter0_s_val_i_reg_260 = 'bx;

assign ashr_ln196_1_fu_1148_p2 = $signed(q_val_4_reg_1882_pp0_iter13_reg) >>> conv_i_i_i36_icast86_fu_1144_p1;

assign ashr_ln196_2_fu_1166_p2 = $signed(q_val_8_reg_1891_pp0_iter13_reg) >>> conv_i_i_i36_icast88_fu_1162_p1;

assign ashr_ln196_3_fu_1184_p2 = $signed(q_val_16_reg_1900_pp0_iter13_reg) >>> conv_i_i_i36_icast90_fu_1180_p1;

assign ashr_ln196_4_fu_1202_p2 = $signed(q_val_1_reg_1909_pp0_iter13_reg) >>> conv_i_i_i36_icast92_fu_1198_p1;

assign ashr_ln196_5_fu_1220_p2 = $signed(q_val_5_reg_1918_pp0_iter13_reg) >>> conv_i_i_i36_icast94_fu_1216_p1;

assign ashr_ln196_6_fu_1238_p2 = $signed(q_val_9_reg_1927_pp0_iter13_reg) >>> conv_i_i_i36_icast96_fu_1234_p1;

assign ashr_ln196_7_fu_1256_p2 = $signed(q_val_2_reg_1936_pp0_iter13_reg) >>> conv_i_i_i36_icast98_fu_1252_p1;

assign ashr_ln196_fu_1130_p2 = $signed(q_val_reg_1873_pp0_iter13_reg) >>> conv_i_i_i36_icast_fu_1126_p1;

assign conv_i_i9_i_cast_i_fu_1111_p1 = $signed(conv_i_i9_i_i_reg_2217);

assign conv_i_i9_i_cast_icast85_fu_1135_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast87_fu_1153_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast89_fu_1171_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast91_fu_1189_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast93_fu_1207_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast95_fu_1225_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast97_fu_1243_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_cast_icast_fu_1117_p1 = conv_i_i9_i_cast_i_fu_1111_p1[26:0];

assign conv_i_i9_i_i_fu_1100_p2 = (5'd1 - s_val_cast_i_fu_1082_p1);

assign conv_i_i_i36_i_fu_1114_p1 = sub_i_i_i_reg_2200;

assign conv_i_i_i36_icast86_fu_1144_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast88_fu_1162_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast90_fu_1180_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast92_fu_1198_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast94_fu_1216_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast96_fu_1234_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast98_fu_1252_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign conv_i_i_i36_icast_fu_1126_p1 = conv_i_i_i36_i_fu_1114_p1[26:0];

assign icmp_ln12_fu_790_p2 = (($signed(abs_max_reg_2073) > $signed(27'd0)) ? 1'b1 : 1'b0);

assign icmp_ln175_fu_455_p2 = ((ap_sig_allocacmp_vec_3 == select_ln175_cast_fu_443_p1) ? 1'b1 : 1'b0);

assign icmp_ln224_1_fu_652_p2 = (($signed(zext_ln185_fu_649_p1) < $signed(select_ln185_2_reg_1995)) ? 1'b1 : 1'b0);

assign icmp_ln224_2_fu_675_p2 = (($signed(select_ln185_3_reg_2001) < $signed(select_ln185_4_reg_2007)) ? 1'b1 : 1'b0);

assign icmp_ln224_3_fu_696_p2 = (($signed(select_ln185_5_reg_2013) < $signed(select_ln185_6_reg_2019)) ? 1'b1 : 1'b0);

assign icmp_ln224_4_fu_717_p2 = (($signed(select_ln185_7_reg_2025) < $signed(select_ln185_8_reg_2031)) ? 1'b1 : 1'b0);

assign icmp_ln224_5_fu_738_p2 = (($signed(select_ln185_9_reg_2037) < $signed(select_ln185_10_reg_2043)) ? 1'b1 : 1'b0);

assign icmp_ln224_6_fu_759_p2 = (($signed(select_ln185_11_reg_2049) < $signed(select_ln185_12_reg_2055)) ? 1'b1 : 1'b0);

assign icmp_ln224_7_fu_780_p2 = (($signed(select_ln185_13_reg_2061) < $signed(select_ln185_14_reg_2067)) ? 1'b1 : 1'b0);

assign icmp_ln224_fu_624_p2 = (($signed(select_ln185_fu_615_p3) > $signed(27'd0)) ? 1'b1 : 1'b0);

assign icmp_ln42_1_fu_1509_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0_1472474484490504514_i_reg_407) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_2_fu_1537_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0_2478480494500518_i_reg_389) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_3_fu_1565_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0_3486488506512_i_reg_416) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_4_fu_1593_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0_4496498520_i_reg_380) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_5_fu_1625_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0_5508510_i_reg_425) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_6_fu_1653_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0_6522_i_reg_371) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_7_fu_1701_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_33_reg_434) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln42_fu_1461_p2 = (($signed(ap_phi_reg_pp0_iter16_q_val_0468470476482492502516_i_reg_398) < $signed(27'd134217600)) ? 1'b1 : 1'b0);

assign icmp_ln43_10_fu_1553_p2 = (($signed(tmp_55_fu_1543_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_11_fu_1581_p2 = (($signed(tmp_56_fu_1571_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_12_fu_1609_p2 = (($signed(tmp_57_fu_1599_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_13_fu_1641_p2 = (($signed(tmp_58_fu_1631_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_14_fu_1669_p2 = (($signed(tmp_59_fu_1659_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_15_fu_1717_p2 = (($signed(tmp_60_fu_1707_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_8_fu_1477_p2 = (($signed(tmp_53_fu_1467_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_9_fu_1525_p2 = (($signed(tmp_54_fu_1515_p4) > $signed(20'd0)) ? 1'b1 : 1'b0);

assign icmp_ln43_fu_1044_p2 = ((tmp_51_fu_1034_p4 == 2'd1) ? 1'b1 : 1'b0);

assign lnot_i_i_i_fu_1106_p2 = ((s_val_reg_2194 == 4'd0) ? 1'b1 : 1'b0);

assign lnot_i_i_i_reg_2222_pp0_iter14_reg = lnot_i_i_i_reg_2222;

assign or_ln189_fu_1068_p2 = (tmp_50_fu_1026_p3 | icmp_ln43_fu_1044_p2);

assign or_ln201_1_fu_1531_p2 = (icmp_ln43_9_fu_1525_p2 | icmp_ln42_1_fu_1509_p2);

assign or_ln201_2_fu_1559_p2 = (icmp_ln43_10_fu_1553_p2 | icmp_ln42_2_fu_1537_p2);

assign or_ln201_3_fu_1587_p2 = (icmp_ln43_11_fu_1581_p2 | icmp_ln42_3_fu_1565_p2);

assign or_ln201_4_fu_1619_p2 = (icmp_ln43_12_fu_1609_p2 | icmp_ln42_4_fu_1593_p2);

assign or_ln201_5_fu_1647_p2 = (icmp_ln43_13_fu_1641_p2 | icmp_ln42_5_fu_1625_p2);

assign or_ln201_6_fu_1687_p2 = (icmp_ln43_14_fu_1669_p2 | icmp_ln42_6_fu_1653_p2);

assign or_ln201_7_fu_1735_p2 = (icmp_ln43_15_fu_1717_p2 | icmp_ln42_7_fu_1701_p2);

assign or_ln201_fu_1495_p2 = (icmp_ln43_8_fu_1477_p2 | icmp_ln42_fu_1461_p2);

assign q_stream_TDATA = {{{{{{{{select_ln201_5_reg_2356}, {select_ln201_3_reg_2351}}, {select_ln201_15_reg_2381}}, {select_ln201_13_reg_2376}}, {select_ln201_11_reg_2371}}, {select_ln201_9_reg_2366}}, {select_ln201_7_reg_2361}}, {select_ln201_1_reg_2346}};

assign q_val_10_fu_1266_p2 = (q_val_6_fu_1261_p3 + 27'd1);

assign q_val_11_fu_1322_p4 = {{q_val_15_fu_1316_p2[26:1]}};

assign q_val_12_fu_1286_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_1_reg_2236 : ashr_ln196_1_reg_2241);

assign q_val_13_fu_1291_p2 = (q_val_12_fu_1286_p3 + 27'd1);

assign q_val_14_fu_1311_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_2_reg_2246 : ashr_ln196_2_reg_2251);

assign q_val_15_fu_1316_p2 = (q_val_14_fu_1311_p3 + 27'd1);

assign q_val_17_fu_1336_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_3_reg_2256 : ashr_ln196_3_reg_2261);

assign q_val_18_fu_1341_p2 = (q_val_17_fu_1336_p3 + 27'd1);

assign q_val_19_fu_1347_p4 = {{q_val_18_fu_1341_p2[26:1]}};

assign q_val_20_fu_1361_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_4_reg_2266 : ashr_ln196_4_reg_2271);

assign q_val_21_fu_1366_p2 = (q_val_20_fu_1361_p3 + 27'd1);

assign q_val_22_fu_1372_p4 = {{q_val_21_fu_1366_p2[26:1]}};

assign q_val_23_fu_1386_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_5_reg_2276 : ashr_ln196_5_reg_2281);

assign q_val_24_fu_1391_p2 = (q_val_23_fu_1386_p3 + 27'd1);

assign q_val_25_fu_1397_p4 = {{q_val_24_fu_1391_p2[26:1]}};

assign q_val_26_fu_1411_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_6_reg_2286 : ashr_ln196_6_reg_2291);

assign q_val_27_fu_1416_p2 = (q_val_26_fu_1411_p3 + 27'd1);

assign q_val_28_fu_1422_p4 = {{q_val_27_fu_1416_p2[26:1]}};

assign q_val_29_fu_1436_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_7_reg_2296 : ashr_ln196_7_reg_2301);

assign q_val_30_fu_1441_p2 = (q_val_29_fu_1436_p3 + 27'd1);

assign q_val_31_fu_1447_p4 = {{q_val_30_fu_1441_p2[26:1]}};

assign q_val_3_fu_1272_p4 = {{q_val_10_fu_1266_p2[26:1]}};

assign q_val_6_fu_1261_p3 = ((tmp_52_reg_2205_pp0_iter14_reg[0:0] == 1'b1) ? shl_ln196_reg_2226 : ashr_ln196_reg_2231);

assign q_val_7_fu_1297_p4 = {{q_val_13_fu_1291_p2[26:1]}};

assign q_val_fu_472_p1 = act_stream_i_dout[26:0];

assign s_stream_TDATA = s_val_reg_2194_pp0_iter16_reg;

assign s_val_cast_i_fu_1082_p1 = s_val_fu_1074_p3;

assign s_val_fu_1074_p3 = ((or_ln189_fu_1068_p2[0:0] == 1'b1) ? select_ln189_fu_1060_p3 : trunc_ln189_fu_1050_p1);

assign select_ln16_fu_1019_p3 = ((trunc_ln10_reg_2080[0:0] == 1'b1) ? 6'd58 : 6'd57);

assign select_ln175_cast_fu_443_p1 = $signed(select_ln175);

assign select_ln185_10_fu_732_p3 = ((tmp_21_reg_1970_pp0_iter6_reg[0:0] == 1'b1) ? sub_ln185_5_fu_727_p2 : q_val_5_reg_1918_pp0_iter6_reg);

assign select_ln185_11_fu_742_p3 = ((icmp_ln224_5_fu_738_p2[0:0] == 1'b1) ? select_ln185_10_reg_2043 : select_ln185_9_reg_2037);

assign select_ln185_12_fu_753_p3 = ((tmp_22_reg_1975_pp0_iter7_reg[0:0] == 1'b1) ? sub_ln185_6_fu_748_p2 : q_val_9_reg_1927_pp0_iter7_reg);

assign select_ln185_13_fu_763_p3 = ((icmp_ln224_6_fu_759_p2[0:0] == 1'b1) ? select_ln185_12_reg_2055 : select_ln185_11_reg_2049);

assign select_ln185_14_fu_774_p3 = ((tmp_23_reg_1980_pp0_iter8_reg[0:0] == 1'b1) ? sub_ln185_7_fu_769_p2 : q_val_2_reg_1936_pp0_iter8_reg);

assign select_ln185_1_fu_630_p3 = ((icmp_ln224_fu_624_p2[0:0] == 1'b1) ? trunc_ln224_fu_620_p1 : 26'd0);

assign select_ln185_2_fu_643_p3 = ((tmp_17_reg_1950_pp0_iter2_reg[0:0] == 1'b1) ? sub_ln185_1_fu_638_p2 : q_val_4_reg_1882_pp0_iter2_reg);

assign select_ln185_3_fu_657_p3 = ((icmp_ln224_1_fu_652_p2[0:0] == 1'b1) ? select_ln185_2_reg_1995 : zext_ln185_fu_649_p1);

assign select_ln185_4_fu_669_p3 = ((tmp_18_reg_1955_pp0_iter3_reg[0:0] == 1'b1) ? sub_ln185_2_fu_664_p2 : q_val_8_reg_1891_pp0_iter3_reg);

assign select_ln185_5_fu_679_p3 = ((icmp_ln224_2_fu_675_p2[0:0] == 1'b1) ? select_ln185_4_reg_2007 : select_ln185_3_reg_2001);

assign select_ln185_6_fu_690_p3 = ((tmp_19_reg_1960_pp0_iter4_reg[0:0] == 1'b1) ? sub_ln185_3_fu_685_p2 : q_val_16_reg_1900_pp0_iter4_reg);

assign select_ln185_7_fu_700_p3 = ((icmp_ln224_3_fu_696_p2[0:0] == 1'b1) ? select_ln185_6_reg_2019 : select_ln185_5_reg_2013);

assign select_ln185_8_fu_711_p3 = ((tmp_20_reg_1965_pp0_iter5_reg[0:0] == 1'b1) ? sub_ln185_4_fu_706_p2 : q_val_1_reg_1909_pp0_iter5_reg);

assign select_ln185_9_fu_721_p3 = ((icmp_ln224_4_fu_717_p2[0:0] == 1'b1) ? select_ln185_8_reg_2031 : select_ln185_7_reg_2025);

assign select_ln185_fu_615_p3 = ((tmp_reg_1945_pp0_iter2_reg[0:0] == 1'b1) ? sub_ln185_reg_1985 : q_val_reg_1873_pp0_iter2_reg);

assign select_ln189_fu_1060_p3 = ((xor_ln189_fu_1054_p2[0:0] == 1'b1) ? 4'd15 : 4'd0);

assign select_ln201_10_fu_1789_p3 = ((icmp_ln42_3_fu_1565_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_11_fu_1801_p3 = ((or_ln201_3_fu_1587_p2[0:0] == 1'b1) ? select_ln201_10_fu_1789_p3 : trunc_ln204_2_fu_1797_p1);

assign select_ln201_12_fu_1809_p3 = ((icmp_ln42_4_fu_1593_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_13_fu_1817_p3 = ((or_ln201_4_fu_1619_p2[0:0] == 1'b1) ? select_ln201_12_fu_1809_p3 : trunc_ln201_1_fu_1615_p1);

assign select_ln201_14_fu_1825_p3 = ((icmp_ln42_5_fu_1625_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_15_fu_1837_p3 = ((or_ln201_5_fu_1647_p2[0:0] == 1'b1) ? select_ln201_14_fu_1825_p3 : trunc_ln204_3_fu_1833_p1);

assign select_ln201_1_fu_1501_p3 = ((or_ln201_fu_1495_p2[0:0] == 1'b1) ? select_ln201_fu_1487_p3 : trunc_ln201_fu_1483_p1);

assign select_ln201_2_fu_1679_p3 = ((icmp_ln42_6_fu_1653_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_3_fu_1693_p3 = ((or_ln201_6_fu_1687_p2[0:0] == 1'b1) ? select_ln201_2_fu_1679_p3 : trunc_ln201_2_fu_1675_p1);

assign select_ln201_4_fu_1727_p3 = ((icmp_ln42_7_fu_1701_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_5_fu_1741_p3 = ((or_ln201_7_fu_1735_p2[0:0] == 1'b1) ? select_ln201_4_fu_1727_p3 : trunc_ln201_3_fu_1723_p1);

assign select_ln201_6_fu_1749_p3 = ((icmp_ln42_1_fu_1509_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_7_fu_1761_p3 = ((or_ln201_1_fu_1531_p2[0:0] == 1'b1) ? select_ln201_6_fu_1749_p3 : trunc_ln204_fu_1757_p1);

assign select_ln201_8_fu_1769_p3 = ((icmp_ln42_2_fu_1537_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign select_ln201_9_fu_1781_p3 = ((or_ln201_2_fu_1559_p2[0:0] == 1'b1) ? select_ln201_8_fu_1769_p3 : trunc_ln204_1_fu_1777_p1);

assign select_ln201_fu_1487_p3 = ((icmp_ln42_fu_1461_p2[0:0] == 1'b1) ? 8'd128 : 8'd127);

assign sext_ln198_1_fu_1307_p1 = $signed(q_val_7_fu_1297_p4);

assign sext_ln198_2_fu_1332_p1 = $signed(q_val_11_fu_1322_p4);

assign sext_ln198_3_fu_1357_p1 = $signed(q_val_19_fu_1347_p4);

assign sext_ln198_4_fu_1382_p1 = $signed(q_val_22_fu_1372_p4);

assign sext_ln198_5_fu_1407_p1 = $signed(q_val_25_fu_1397_p4);

assign sext_ln198_6_fu_1432_p1 = $signed(q_val_28_fu_1422_p4);

assign sext_ln198_7_fu_1457_p1 = $signed(q_val_31_fu_1447_p4);

assign sext_ln198_fu_1282_p1 = $signed(q_val_3_fu_1272_p4);

assign shl_ln196_1_fu_1139_p2 = q_val_4_reg_1882_pp0_iter13_reg << conv_i_i9_i_cast_icast85_fu_1135_p1;

assign shl_ln196_2_fu_1157_p2 = q_val_8_reg_1891_pp0_iter13_reg << conv_i_i9_i_cast_icast87_fu_1153_p1;

assign shl_ln196_3_fu_1175_p2 = q_val_16_reg_1900_pp0_iter13_reg << conv_i_i9_i_cast_icast89_fu_1171_p1;

assign shl_ln196_4_fu_1193_p2 = q_val_1_reg_1909_pp0_iter13_reg << conv_i_i9_i_cast_icast91_fu_1189_p1;

assign shl_ln196_5_fu_1211_p2 = q_val_5_reg_1918_pp0_iter13_reg << conv_i_i9_i_cast_icast93_fu_1207_p1;

assign shl_ln196_6_fu_1229_p2 = q_val_9_reg_1927_pp0_iter13_reg << conv_i_i9_i_cast_icast95_fu_1225_p1;

assign shl_ln196_7_fu_1247_p2 = q_val_2_reg_1936_pp0_iter13_reg << conv_i_i9_i_cast_icast97_fu_1243_p1;

assign shl_ln196_fu_1121_p2 = q_val_reg_1873_pp0_iter13_reg << conv_i_i9_i_cast_icast_fu_1117_p1;

assign sub_i_i_i_fu_1086_p2 = ($signed(s_val_cast_i_fu_1082_p1) + $signed(5'd31));

assign sub_ln185_1_fu_638_p2 = (27'd0 - q_val_4_reg_1882_pp0_iter2_reg);

assign sub_ln185_2_fu_664_p2 = (27'd0 - q_val_8_reg_1891_pp0_iter3_reg);

assign sub_ln185_3_fu_685_p2 = (27'd0 - q_val_16_reg_1900_pp0_iter4_reg);

assign sub_ln185_4_fu_706_p2 = (27'd0 - q_val_1_reg_1909_pp0_iter5_reg);

assign sub_ln185_5_fu_727_p2 = (27'd0 - q_val_5_reg_1918_pp0_iter6_reg);

assign sub_ln185_6_fu_748_p2 = (27'd0 - q_val_9_reg_1927_pp0_iter7_reg);

assign sub_ln185_7_fu_769_p2 = (27'd0 - q_val_2_reg_1936_pp0_iter8_reg);

assign sub_ln185_fu_610_p2 = (27'd0 - q_val_reg_1873);

assign tmp_24_reg_2085_pp0_iter11_reg = tmp_24_reg_2085;

assign tmp_25_reg_2089_pp0_iter11_reg = tmp_25_reg_2089;

assign tmp_26_reg_2093_pp0_iter11_reg = tmp_26_reg_2093;

assign tmp_27_reg_2097_pp0_iter11_reg = tmp_27_reg_2097;

assign tmp_28_reg_2101_pp0_iter11_reg = tmp_28_reg_2101;

assign tmp_29_reg_2105_pp0_iter11_reg = tmp_29_reg_2105;

assign tmp_30_reg_2109_pp0_iter11_reg = tmp_30_reg_2109;

assign tmp_31_reg_2113_pp0_iter11_reg = tmp_31_reg_2113;

assign tmp_32_reg_2117_pp0_iter11_reg = tmp_32_reg_2117;

assign tmp_33_reg_2121_pp0_iter11_reg = tmp_33_reg_2121;

assign tmp_34_reg_2125_pp0_iter11_reg = tmp_34_reg_2125;

assign tmp_35_reg_2129_pp0_iter11_reg = tmp_35_reg_2129;

assign tmp_36_reg_2133_pp0_iter11_reg = tmp_36_reg_2133;

assign tmp_37_reg_2137_pp0_iter11_reg = tmp_37_reg_2137;

assign tmp_38_reg_2141_pp0_iter11_reg = tmp_38_reg_2141;

assign tmp_39_reg_2145_pp0_iter11_reg = tmp_39_reg_2145;

assign tmp_40_reg_2149_pp0_iter11_reg = tmp_40_reg_2149;

assign tmp_41_reg_2153_pp0_iter11_reg = tmp_41_reg_2153;

assign tmp_42_reg_2157_pp0_iter11_reg = tmp_42_reg_2157;

assign tmp_43_reg_2161_pp0_iter11_reg = tmp_43_reg_2161;

assign tmp_44_reg_2165_pp0_iter11_reg = tmp_44_reg_2165;

assign tmp_45_reg_2169_pp0_iter11_reg = tmp_45_reg_2169;

assign tmp_46_reg_2173_pp0_iter11_reg = tmp_46_reg_2173;

assign tmp_47_reg_2177_pp0_iter11_reg = tmp_47_reg_2177;

assign tmp_48_reg_2181_pp0_iter11_reg = tmp_48_reg_2181;

assign tmp_49_reg_2185_pp0_iter11_reg = tmp_49_reg_2185;

assign tmp_50_fu_1026_p3 = ap_phi_reg_pp0_iter13_s_val_i_reg_260[32'd5];

assign tmp_51_fu_1034_p4 = {{ap_phi_reg_pp0_iter13_s_val_i_reg_260[5:4]}};

assign tmp_53_fu_1467_p4 = {{ap_phi_reg_pp0_iter16_q_val_0468470476482492502516_i_reg_398[26:7]}};

assign tmp_54_fu_1515_p4 = {{ap_phi_reg_pp0_iter16_q_val_0_1472474484490504514_i_reg_407[26:7]}};

assign tmp_55_fu_1543_p4 = {{ap_phi_reg_pp0_iter16_q_val_0_2478480494500518_i_reg_389[26:7]}};

assign tmp_56_fu_1571_p4 = {{ap_phi_reg_pp0_iter16_q_val_0_3486488506512_i_reg_416[26:7]}};

assign tmp_57_fu_1599_p4 = {{ap_phi_reg_pp0_iter16_q_val_0_4496498520_i_reg_380[26:7]}};

assign tmp_58_fu_1631_p4 = {{ap_phi_reg_pp0_iter16_q_val_0_5508510_i_reg_425[26:7]}};

assign tmp_59_fu_1659_p4 = {{ap_phi_reg_pp0_iter16_q_val_0_6522_i_reg_371[26:7]}};

assign tmp_60_fu_1707_p4 = {{ap_phi_reg_pp0_iter16_q_val_33_reg_434[26:7]}};

assign trunc_ln10_fu_807_p1 = x_16_fu_800_p3[0:0];

assign trunc_ln189_fu_1050_p1 = ap_phi_reg_pp0_iter13_s_val_i_reg_260[3:0];

assign trunc_ln201_1_fu_1615_p1 = ap_phi_reg_pp0_iter16_q_val_0_4496498520_i_reg_380[7:0];

assign trunc_ln201_2_fu_1675_p1 = ap_phi_reg_pp0_iter16_q_val_0_6522_i_reg_371[7:0];

assign trunc_ln201_3_fu_1723_p1 = ap_phi_reg_pp0_iter16_q_val_33_reg_434[7:0];

assign trunc_ln201_fu_1483_p1 = ap_phi_reg_pp0_iter16_q_val_0468470476482492502516_i_reg_398[7:0];

assign trunc_ln204_1_fu_1777_p1 = ap_phi_reg_pp0_iter16_q_val_0_2478480494500518_i_reg_389[7:0];

assign trunc_ln204_2_fu_1797_p1 = ap_phi_reg_pp0_iter16_q_val_0_3486488506512_i_reg_416[7:0];

assign trunc_ln204_3_fu_1833_p1 = ap_phi_reg_pp0_iter16_q_val_0_5508510_i_reg_425[7:0];

assign trunc_ln204_fu_1757_p1 = ap_phi_reg_pp0_iter16_q_val_0_1472474484490504514_i_reg_407[7:0];

assign trunc_ln224_fu_620_p1 = select_ln185_fu_615_p3[25:0];

assign vec_4_fu_461_p2 = (ap_sig_allocacmp_vec_3 + 19'd1);

assign x_15_fu_795_p2 = ($signed(abs_max_reg_2073) + $signed(27'd134217727));

assign x_16_fu_800_p3 = ((icmp_ln12_fu_790_p2[0:0] == 1'b1) ? x_15_fu_795_p2 : abs_max_reg_2073);

assign xor_ln189_fu_1054_p2 = (tmp_50_fu_1026_p3 ^ 1'd1);

assign zext_ln185_fu_649_p1 = select_ln185_1_reg_1990;

endmodule //SILU_GELU_shared_mlp_quantize4_Pipeline_VITIS_LOOP_175_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_shared_activation_to_common3 (
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
        mode,
        mlp_i_stream_TDATA,
        mlp_i_stream_TVALID,
        mlp_i_stream_TREADY,
        act_stream_din,
        act_stream_num_data_valid,
        act_stream_fifo_cap,
        act_stream_full_n,
        act_stream_write,
        mode_c_din,
        mode_c_num_data_valid,
        mode_c_fifo_cap,
        mode_c_full_n,
        mode_c_write
);

parameter    ap_ST_fsm_state1 = 2'd1;
parameter    ap_ST_fsm_state2 = 2'd2;

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
input  [0:0] mode;
input  [223:0] mlp_i_stream_TDATA;
input   mlp_i_stream_TVALID;
output   mlp_i_stream_TREADY;
output  [215:0] act_stream_din;
input  [5:0] act_stream_num_data_valid;
input  [5:0] act_stream_fifo_cap;
input   act_stream_full_n;
output   act_stream_write;
output  [0:0] mode_c_din;
input  [2:0] mode_c_num_data_valid;
input  [2:0] mode_c_fifo_cap;
input   mode_c_full_n;
output   mode_c_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg mlp_i_stream_TREADY;
reg[215:0] act_stream_din;
reg act_stream_write;
reg mode_c_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [1:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    mode_c_blk_n;
reg    ap_block_state1;
wire   [215:0] grp_llm_silu_em_to_common_fu_60_act_stream_din;
wire    grp_llm_silu_em_to_common_fu_60_act_stream_write;
wire    grp_llm_silu_em_to_common_fu_60_mlp_i_stream_TREADY;
wire    grp_llm_silu_em_to_common_fu_60_ap_start;
wire    grp_llm_silu_em_to_common_fu_60_ap_done;
wire    grp_llm_silu_em_to_common_fu_60_ap_ready;
wire    grp_llm_silu_em_to_common_fu_60_ap_idle;
reg    grp_llm_silu_em_to_common_fu_60_ap_continue;
wire    grp_vit_gelu_to_common_fu_70_ap_start;
wire    grp_vit_gelu_to_common_fu_70_ap_done;
wire    grp_vit_gelu_to_common_fu_70_ap_idle;
wire    grp_vit_gelu_to_common_fu_70_ap_ready;
wire   [215:0] grp_vit_gelu_to_common_fu_70_act_stream_din;
wire    grp_vit_gelu_to_common_fu_70_act_stream_write;
wire    grp_vit_gelu_to_common_fu_70_mlp_i_stream_TREADY;
reg    grp_llm_silu_em_to_common_fu_60_ap_start_reg;
reg    ap_block_state1_ignore_call0;
wire    ap_CS_fsm_state2;
wire    ap_sync_grp_llm_silu_em_to_common_fu_60_ap_ready;
wire    ap_sync_grp_llm_silu_em_to_common_fu_60_ap_done;
reg    ap_block_state2_on_subcall_done;
reg    ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_ready;
reg    ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_done;
reg    grp_vit_gelu_to_common_fu_70_ap_start_reg;
reg   [1:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 start_once_reg = 1'b0;
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 2'd1;
//#0 grp_llm_silu_em_to_common_fu_60_ap_start_reg = 1'b0;
//#0 ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_ready = 1'b0;
//#0 ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_done = 1'b0;
//#0 grp_vit_gelu_to_common_fu_70_ap_start_reg = 1'b0;
end

SILU_GELU_llm_silu_em_to_common grp_llm_silu_em_to_common_fu_60(
    .mlp_i_stream_TDATA(mlp_i_stream_TDATA),
    .act_stream_din(grp_llm_silu_em_to_common_fu_60_act_stream_din),
    .act_stream_full_n(act_stream_full_n),
    .act_stream_write(grp_llm_silu_em_to_common_fu_60_act_stream_write),
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .mlp_i_stream_TVALID(mlp_i_stream_TVALID),
    .mlp_i_stream_TREADY(grp_llm_silu_em_to_common_fu_60_mlp_i_stream_TREADY),
    .ap_start(grp_llm_silu_em_to_common_fu_60_ap_start),
    .ap_done(grp_llm_silu_em_to_common_fu_60_ap_done),
    .ap_ready(grp_llm_silu_em_to_common_fu_60_ap_ready),
    .ap_idle(grp_llm_silu_em_to_common_fu_60_ap_idle),
    .ap_continue(grp_llm_silu_em_to_common_fu_60_ap_continue)
);

SILU_GELU_vit_gelu_to_common grp_vit_gelu_to_common_fu_70(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_vit_gelu_to_common_fu_70_ap_start),
    .ap_done(grp_vit_gelu_to_common_fu_70_ap_done),
    .ap_idle(grp_vit_gelu_to_common_fu_70_ap_idle),
    .ap_ready(grp_vit_gelu_to_common_fu_70_ap_ready),
    .mlp_i_stream_TVALID(mlp_i_stream_TVALID),
    .act_stream_din(grp_vit_gelu_to_common_fu_70_act_stream_din),
    .act_stream_num_data_valid(6'd0),
    .act_stream_fifo_cap(6'd0),
    .act_stream_full_n(act_stream_full_n),
    .act_stream_write(grp_vit_gelu_to_common_fu_70_act_stream_write),
    .mlp_i_stream_TDATA(mlp_i_stream_TDATA),
    .mlp_i_stream_TREADY(grp_vit_gelu_to_common_fu_70_mlp_i_stream_TREADY)
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
        end else if (((1'b0 == ap_block_state2_on_subcall_done) & (1'b1 == ap_CS_fsm_state2))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_done <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state2_on_subcall_done) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
            ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_done <= 1'b0;
        end else if ((grp_llm_silu_em_to_common_fu_60_ap_done == 1'b1)) begin
            ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_done <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_ready <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state2_on_subcall_done) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
            ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_ready <= 1'b0;
        end else if ((grp_llm_silu_em_to_common_fu_60_ap_ready == 1'b1)) begin
            ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_ready <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_llm_silu_em_to_common_fu_60_ap_start_reg <= 1'b0;
    end else begin
        if ((((1'b0 == ap_block_state1_ignore_call0) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state1)) | ((mode == 1'd0) & (1'b1 == ap_CS_fsm_state2) & (ap_sync_grp_llm_silu_em_to_common_fu_60_ap_ready == 1'b0)))) begin
            grp_llm_silu_em_to_common_fu_60_ap_start_reg <= 1'b1;
        end else if ((grp_llm_silu_em_to_common_fu_60_ap_ready == 1'b1)) begin
            grp_llm_silu_em_to_common_fu_60_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_vit_gelu_to_common_fu_70_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state1_ignore_call0) & (mode == 1'd1) & (1'b1 == ap_CS_fsm_state1))) begin
            grp_vit_gelu_to_common_fu_70_ap_start_reg <= 1'b1;
        end else if ((grp_vit_gelu_to_common_fu_70_ap_ready == 1'b1)) begin
            grp_vit_gelu_to_common_fu_70_ap_start_reg <= 1'b0;
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
    if ((1'b1 == ap_CS_fsm_state2)) begin
        if ((mode == 1'd1)) begin
            act_stream_din = grp_vit_gelu_to_common_fu_70_act_stream_din;
        end else if ((mode == 1'd0)) begin
            act_stream_din = grp_llm_silu_em_to_common_fu_60_act_stream_din;
        end else begin
            act_stream_din = grp_vit_gelu_to_common_fu_70_act_stream_din;
        end
    end else begin
        act_stream_din = grp_vit_gelu_to_common_fu_70_act_stream_din;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        if ((mode == 1'd1)) begin
            act_stream_write = grp_vit_gelu_to_common_fu_70_act_stream_write;
        end else if ((mode == 1'd0)) begin
            act_stream_write = grp_llm_silu_em_to_common_fu_60_act_stream_write;
        end else begin
            act_stream_write = 1'b0;
        end
    end else begin
        act_stream_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state2_on_subcall_done)) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state2_on_subcall_done) & (1'b1 == ap_CS_fsm_state2))) begin
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
    if (((1'b0 == ap_block_state2_on_subcall_done) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        grp_llm_silu_em_to_common_fu_60_ap_continue = 1'b1;
    end else begin
        grp_llm_silu_em_to_common_fu_60_ap_continue = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state2_on_subcall_done) & (1'b1 == ap_CS_fsm_state2))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        if ((mode == 1'd1)) begin
            mlp_i_stream_TREADY = grp_vit_gelu_to_common_fu_70_mlp_i_stream_TREADY;
        end else if ((mode == 1'd0)) begin
            mlp_i_stream_TREADY = grp_llm_silu_em_to_common_fu_60_mlp_i_stream_TREADY;
        end else begin
            mlp_i_stream_TREADY = 1'b0;
        end
    end else begin
        mlp_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        mode_c_blk_n = mode_c_full_n;
    end else begin
        mode_c_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        mode_c_write = 1'b1;
    end else begin
        mode_c_write = 1'b0;
    end
end

always @ (*) begin
    if (((start_full_n == 1'b0) & (start_once_reg == 1'b0))) begin
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
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((1'b0 == ap_block_state2_on_subcall_done) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (mode_c_full_n == 1'b0) | (ap_done_reg == 1'b1));
end

always @ (*) begin
    ap_block_state1_ignore_call0 = ((real_start == 1'b0) | (mode_c_full_n == 1'b0) | (ap_done_reg == 1'b1));
end

always @ (*) begin
    ap_block_state2_on_subcall_done = (((grp_vit_gelu_to_common_fu_70_ap_done == 1'b0) & (mode == 1'd1)) | ((mode == 1'd0) & ((ap_sync_grp_llm_silu_em_to_common_fu_60_ap_ready & ap_sync_grp_llm_silu_em_to_common_fu_60_ap_done) == 1'b0)));
end

assign ap_ready = internal_ap_ready;

assign ap_sync_grp_llm_silu_em_to_common_fu_60_ap_done = (grp_llm_silu_em_to_common_fu_60_ap_done | ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_done);

assign ap_sync_grp_llm_silu_em_to_common_fu_60_ap_ready = (grp_llm_silu_em_to_common_fu_60_ap_ready | ap_sync_reg_grp_llm_silu_em_to_common_fu_60_ap_ready);

assign grp_llm_silu_em_to_common_fu_60_ap_start = grp_llm_silu_em_to_common_fu_60_ap_start_reg;

assign grp_vit_gelu_to_common_fu_70_ap_start = grp_vit_gelu_to_common_fu_70_ap_start_reg;

assign mode_c_din = mode;

assign start_out = real_start;

endmodule //SILU_GELU_shared_activation_to_common3
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_llm_silu_em_to_common (
        mlp_i_stream_TDATA,
        act_stream_din,
        act_stream_full_n,
        act_stream_write,
        ap_clk,
        ap_rst,
        mlp_i_stream_TVALID,
        mlp_i_stream_TREADY,
        ap_start,
        ap_done,
        ap_ready,
        ap_idle,
        ap_continue
);


input  [223:0] mlp_i_stream_TDATA;
output  [215:0] act_stream_din;
input   act_stream_full_n;
output   act_stream_write;
input   ap_clk;
input   ap_rst;
input   mlp_i_stream_TVALID;
output   mlp_i_stream_TREADY;
input   ap_start;
output   ap_done;
output   ap_ready;
output   ap_idle;
input   ap_continue;

wire    llm_split_ug_U0_ap_start;
wire    llm_split_ug_U0_start_full_n;
wire    llm_split_ug_U0_ap_done;
wire    llm_split_ug_U0_ap_continue;
wire    llm_split_ug_U0_ap_idle;
wire    llm_split_ug_U0_ap_ready;
wire   [167:0] llm_split_ug_U0_g_stream_din;
wire    llm_split_ug_U0_g_stream_write;
wire   [167:0] llm_split_ug_U0_u_stream_din;
wire    llm_split_ug_U0_u_stream_write;
wire    llm_split_ug_U0_start_out;
wire    llm_split_ug_U0_start_write;
wire    llm_split_ug_U0_mlp_i_stream_TREADY;
wire    do_silu_U0_g_stream_read;
wire   [167:0] do_silu_U0_silu_stream_din;
wire    do_silu_U0_silu_stream_write;
wire    do_silu_U0_ap_start;
wire    do_silu_U0_ap_done;
wire    do_silu_U0_ap_ready;
wire    do_silu_U0_ap_idle;
wire    do_silu_U0_ap_continue;
wire    llm_merge_silu_u_U0_ap_start;
wire    llm_merge_silu_u_U0_ap_done;
wire    llm_merge_silu_u_U0_ap_continue;
wire    llm_merge_silu_u_U0_ap_idle;
wire    llm_merge_silu_u_U0_ap_ready;
wire    llm_merge_silu_u_U0_u_stream_read;
wire    llm_merge_silu_u_U0_silu_stream_read;
wire   [215:0] llm_merge_silu_u_U0_act_stream_din;
wire    llm_merge_silu_u_U0_act_stream_write;
wire    u_stream_full_n;
wire   [167:0] u_stream_dout;
wire   [4:0] u_stream_num_data_valid;
wire   [4:0] u_stream_fifo_cap;
wire    u_stream_empty_n;
wire    g_stream_full_n;
wire   [167:0] g_stream_dout;
wire   [4:0] g_stream_num_data_valid;
wire   [4:0] g_stream_fifo_cap;
wire    g_stream_empty_n;
wire    silu_stream_full_n;
wire   [167:0] silu_stream_dout;
wire   [4:0] silu_stream_num_data_valid;
wire   [4:0] silu_stream_fifo_cap;
wire    silu_stream_empty_n;
wire   [0:0] start_for_do_silu_U0_din;
wire    start_for_do_silu_U0_full_n;
wire   [0:0] start_for_do_silu_U0_dout;
wire    start_for_do_silu_U0_empty_n;
wire   [0:0] start_for_llm_merge_silu_u_U0_din;
wire    start_for_llm_merge_silu_u_U0_full_n;
wire   [0:0] start_for_llm_merge_silu_u_U0_dout;
wire    start_for_llm_merge_silu_u_U0_empty_n;

SILU_GELU_llm_split_ug llm_split_ug_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(llm_split_ug_U0_ap_start),
    .start_full_n(llm_split_ug_U0_start_full_n),
    .ap_done(llm_split_ug_U0_ap_done),
    .ap_continue(llm_split_ug_U0_ap_continue),
    .ap_idle(llm_split_ug_U0_ap_idle),
    .ap_ready(llm_split_ug_U0_ap_ready),
    .mlp_i_stream_TVALID(mlp_i_stream_TVALID),
    .g_stream_din(llm_split_ug_U0_g_stream_din),
    .g_stream_num_data_valid(g_stream_num_data_valid),
    .g_stream_fifo_cap(g_stream_fifo_cap),
    .g_stream_full_n(g_stream_full_n),
    .g_stream_write(llm_split_ug_U0_g_stream_write),
    .u_stream_din(llm_split_ug_U0_u_stream_din),
    .u_stream_num_data_valid(u_stream_num_data_valid),
    .u_stream_fifo_cap(u_stream_fifo_cap),
    .u_stream_full_n(u_stream_full_n),
    .u_stream_write(llm_split_ug_U0_u_stream_write),
    .start_out(llm_split_ug_U0_start_out),
    .start_write(llm_split_ug_U0_start_write),
    .mlp_i_stream_TDATA(mlp_i_stream_TDATA),
    .mlp_i_stream_TREADY(llm_split_ug_U0_mlp_i_stream_TREADY)
);

SILU_GELU_do_silu do_silu_U0(
    .g_stream_dout(g_stream_dout),
    .g_stream_empty_n(g_stream_empty_n),
    .g_stream_read(do_silu_U0_g_stream_read),
    .silu_stream_din(do_silu_U0_silu_stream_din),
    .silu_stream_full_n(silu_stream_full_n),
    .silu_stream_write(do_silu_U0_silu_stream_write),
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(do_silu_U0_ap_start),
    .ap_done(do_silu_U0_ap_done),
    .ap_ready(do_silu_U0_ap_ready),
    .ap_idle(do_silu_U0_ap_idle),
    .ap_continue(do_silu_U0_ap_continue)
);

SILU_GELU_llm_merge_silu_u llm_merge_silu_u_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(llm_merge_silu_u_U0_ap_start),
    .ap_done(llm_merge_silu_u_U0_ap_done),
    .ap_continue(llm_merge_silu_u_U0_ap_continue),
    .ap_idle(llm_merge_silu_u_U0_ap_idle),
    .ap_ready(llm_merge_silu_u_U0_ap_ready),
    .u_stream_dout(u_stream_dout),
    .u_stream_num_data_valid(u_stream_num_data_valid),
    .u_stream_fifo_cap(u_stream_fifo_cap),
    .u_stream_empty_n(u_stream_empty_n),
    .u_stream_read(llm_merge_silu_u_U0_u_stream_read),
    .silu_stream_dout(silu_stream_dout),
    .silu_stream_num_data_valid(silu_stream_num_data_valid),
    .silu_stream_fifo_cap(silu_stream_fifo_cap),
    .silu_stream_empty_n(silu_stream_empty_n),
    .silu_stream_read(llm_merge_silu_u_U0_silu_stream_read),
    .act_stream_din(llm_merge_silu_u_U0_act_stream_din),
    .act_stream_num_data_valid(6'd0),
    .act_stream_fifo_cap(6'd0),
    .act_stream_full_n(act_stream_full_n),
    .act_stream_write(llm_merge_silu_u_U0_act_stream_write)
);

SILU_GELU_fifo_w168_d16_A u_stream_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(llm_split_ug_U0_u_stream_din),
    .if_full_n(u_stream_full_n),
    .if_write(llm_split_ug_U0_u_stream_write),
    .if_dout(u_stream_dout),
    .if_num_data_valid(u_stream_num_data_valid),
    .if_fifo_cap(u_stream_fifo_cap),
    .if_empty_n(u_stream_empty_n),
    .if_read(llm_merge_silu_u_U0_u_stream_read)
);

SILU_GELU_fifo_w168_d16_A g_stream_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(llm_split_ug_U0_g_stream_din),
    .if_full_n(g_stream_full_n),
    .if_write(llm_split_ug_U0_g_stream_write),
    .if_dout(g_stream_dout),
    .if_num_data_valid(g_stream_num_data_valid),
    .if_fifo_cap(g_stream_fifo_cap),
    .if_empty_n(g_stream_empty_n),
    .if_read(do_silu_U0_g_stream_read)
);

SILU_GELU_fifo_w168_d16_A silu_stream_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(do_silu_U0_silu_stream_din),
    .if_full_n(silu_stream_full_n),
    .if_write(do_silu_U0_silu_stream_write),
    .if_dout(silu_stream_dout),
    .if_num_data_valid(silu_stream_num_data_valid),
    .if_fifo_cap(silu_stream_fifo_cap),
    .if_empty_n(silu_stream_empty_n),
    .if_read(llm_merge_silu_u_U0_silu_stream_read)
);

SILU_GELU_start_for_do_silu_U0 start_for_do_silu_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_do_silu_U0_din),
    .if_full_n(start_for_do_silu_U0_full_n),
    .if_write(llm_split_ug_U0_start_write),
    .if_dout(start_for_do_silu_U0_dout),
    .if_empty_n(start_for_do_silu_U0_empty_n),
    .if_read(do_silu_U0_ap_ready)
);

SILU_GELU_start_for_llm_merge_silu_u_U0 start_for_llm_merge_silu_u_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_llm_merge_silu_u_U0_din),
    .if_full_n(start_for_llm_merge_silu_u_U0_full_n),
    .if_write(llm_split_ug_U0_start_write),
    .if_dout(start_for_llm_merge_silu_u_U0_dout),
    .if_empty_n(start_for_llm_merge_silu_u_U0_empty_n),
    .if_read(llm_merge_silu_u_U0_ap_ready)
);

assign act_stream_din = llm_merge_silu_u_U0_act_stream_din;

assign act_stream_write = llm_merge_silu_u_U0_act_stream_write;

assign ap_done = llm_merge_silu_u_U0_ap_done;

assign ap_idle = (llm_split_ug_U0_ap_idle & llm_merge_silu_u_U0_ap_idle & do_silu_U0_ap_idle);

assign ap_ready = llm_split_ug_U0_ap_ready;

assign do_silu_U0_ap_continue = 1'b1;

assign do_silu_U0_ap_start = start_for_do_silu_U0_empty_n;

assign llm_merge_silu_u_U0_ap_continue = ap_continue;

assign llm_merge_silu_u_U0_ap_start = start_for_llm_merge_silu_u_U0_empty_n;

assign llm_split_ug_U0_ap_continue = 1'b1;

assign llm_split_ug_U0_ap_start = ap_start;

assign llm_split_ug_U0_start_full_n = (start_for_llm_merge_silu_u_U0_full_n & start_for_do_silu_U0_full_n);

assign mlp_i_stream_TREADY = llm_split_ug_U0_mlp_i_stream_TREADY;

assign start_for_do_silu_U0_din = 1'b1;

assign start_for_llm_merge_silu_u_U0_din = 1'b1;

endmodule //SILU_GELU_llm_silu_em_to_common
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module SILU_GELU_do_silu_parallel (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        adpt_stream_dout,
        adpt_stream_num_data_valid,
        adpt_stream_fifo_cap,
        adpt_stream_empty_n,
        adpt_stream_read,
        silu_stream_din,
        silu_stream_num_data_valid,
        silu_stream_fifo_cap,
        silu_stream_full_n,
        silu_stream_write,
        start_out,
        start_write
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter4_fsm_state5 = 2'd2;
parameter    ap_ST_iter5_fsm_state6 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [20:0] adpt_stream_dout;
input  [2:0] adpt_stream_num_data_valid;
input  [2:0] adpt_stream_fifo_cap;
input   adpt_stream_empty_n;
output   adpt_stream_read;
output  [20:0] silu_stream_din;
input  [2:0] silu_stream_num_data_valid;
input  [2:0] silu_stream_fifo_cap;
input   silu_stream_full_n;
output   silu_stream_write;
output   start_out;
output   start_write;

reg ap_idle;
reg adpt_stream_read;
reg silu_stream_write;
reg start_write;

reg    real_start;
reg    start_once_reg;
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
wire    internal_ap_ready;
wire   [0:0] icmp_ln43_fu_104_p2;
reg    ap_done_reg;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
reg   [0:0] icmp_ln43_reg_229;
reg   [0:0] icmp_ln43_reg_229_pp0_iter4_reg;
reg    ap_block_state6_pp0_stage0_iter5;
wire    ap_CS_iter5_fsm_state6;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
wire   [8:0] SILU_TABLE_address0;
reg    SILU_TABLE_ce0;
wire   [7:0] SILU_TABLE_q0;
reg    adpt_stream_blk_n;
reg    silu_stream_blk_n;
reg   [0:0] icmp_ln43_reg_229_pp0_iter1_reg;
reg   [0:0] icmp_ln43_reg_229_pp0_iter2_reg;
reg   [0:0] icmp_ln43_reg_229_pp0_iter3_reg;
reg   [20:0] adpt_stream_read_reg_233;
reg   [20:0] adpt_stream_read_reg_233_pp0_iter1_reg;
reg   [20:0] adpt_stream_read_reg_233_pp0_iter2_reg;
reg   [20:0] adpt_stream_read_reg_233_pp0_iter3_reg;
reg   [20:0] adpt_stream_read_reg_233_pp0_iter4_reg;
reg   [0:0] tmp_61_reg_239;
reg   [0:0] tmp_61_reg_239_pp0_iter1_reg;
reg   [0:0] tmp_61_reg_239_pp0_iter2_reg;
reg   [0:0] tmp_61_reg_239_pp0_iter3_reg;
reg   [0:0] tmp_61_reg_239_pp0_iter4_reg;
reg   [13:0] tmp_reg_245;
wire   [8:0] trunc_ln55_1_fu_160_p1;
reg   [8:0] trunc_ln55_1_reg_250;
reg   [4:0] tmp_62_reg_255;
wire   [8:0] LUT_IDX_fu_179_p3;
reg   [8:0] LUT_IDX_reg_260;
reg   [7:0] DIFF_reg_270;
wire   [63:0] zext_ln57_fu_186_p1;
reg   [14:0] indvar_flatten_fu_66;
wire   [14:0] add_ln43_fu_110_p2;
wire    ap_loop_init;
reg   [14:0] ap_sig_allocacmp_indvar_flatten_load;
wire   [20:0] sub_ln61_fu_139_p2;
wire   [13:0] tmp_s_fu_144_p4;
wire   [13:0] select_ln61_fu_154_p3;
wire   [0:0] icmp_ln43_16_fu_174_p2;
wire   [19:0] trunc_ln55_fu_190_p1;
wire   [19:0] RELU_fu_193_p3;
wire   [11:0] shl_ln_fu_204_p3;
wire   [20:0] zext_ln58_fu_200_p1;
wire   [20:0] zext_ln59_fu_211_p1;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
reg    ap_ST_iter5_fsm_state6_blk;
wire    ap_start_int;
reg    ap_condition_134;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 start_once_reg = 1'b0;
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 ap_CS_iter4_fsm = 2'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 ap_done_reg = 1'b0;
//#0 indvar_flatten_fu_66 = 15'd0;
end

SILU_GELU_do_silu_parallel_SILU_TABLE_ROM_AUTO_1R #(
    .DataWidth( 8 ),
    .AddressRange( 512 ),
    .AddressWidth( 9 ))
SILU_TABLE_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(SILU_TABLE_address0),
    .ce0(SILU_TABLE_ce0),
    .q0(SILU_TABLE_q0)
);

SILU_GELU_flow_control_loop_pipe flow_control_loop_pipe_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(real_start),
    .ap_ready(internal_ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int),
    .ap_continue(ap_continue)
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
        end else if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((internal_ap_ready == 1'b0) & (real_start == 1'b1))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_134)) begin
        if ((icmp_ln43_fu_104_p2 == 1'd0)) begin
            indvar_flatten_fu_66 <= add_ln43_fu_110_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_66 <= 15'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        DIFF_reg_270 <= SILU_TABLE_q0;
        adpt_stream_read_reg_233_pp0_iter4_reg <= adpt_stream_read_reg_233_pp0_iter3_reg;
        icmp_ln43_reg_229_pp0_iter4_reg <= icmp_ln43_reg_229_pp0_iter3_reg;
        tmp_61_reg_239_pp0_iter4_reg <= tmp_61_reg_239_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        LUT_IDX_reg_260 <= LUT_IDX_fu_179_p3;
        adpt_stream_read_reg_233_pp0_iter2_reg <= adpt_stream_read_reg_233_pp0_iter1_reg;
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln43_reg_229_pp0_iter2_reg <= icmp_ln43_reg_229_pp0_iter1_reg;
        tmp_61_reg_239_pp0_iter2_reg <= tmp_61_reg_239_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        adpt_stream_read_reg_233 <= adpt_stream_dout;
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln43_reg_229 <= icmp_ln43_fu_104_p2;
        tmp_61_reg_239 <= adpt_stream_dout[32'd20];
        tmp_reg_245 <= {{adpt_stream_dout[20:7]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        adpt_stream_read_reg_233_pp0_iter1_reg <= adpt_stream_read_reg_233;
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln43_reg_229_pp0_iter1_reg <= icmp_ln43_reg_229;
        tmp_61_reg_239_pp0_iter1_reg <= tmp_61_reg_239;
        tmp_62_reg_255 <= {{select_ln61_fu_154_p3[13:9]}};
        trunc_ln55_1_reg_250 <= trunc_ln55_1_fu_160_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        adpt_stream_read_reg_233_pp0_iter3_reg <= adpt_stream_read_reg_233_pp0_iter2_reg;
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln43_reg_229_pp0_iter3_reg <= icmp_ln43_reg_229_pp0_iter2_reg;
        tmp_61_reg_239_pp0_iter3_reg <= tmp_61_reg_239_pp0_iter2_reg;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        SILU_TABLE_ce0 = 1'b1;
    end else begin
        SILU_TABLE_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (ap_start_int == 1'b0)) & (icmp_ln43_fu_104_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        adpt_stream_blk_n = adpt_stream_empty_n;
    end else begin
        adpt_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (icmp_ln43_fu_104_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        adpt_stream_read = 1'b1;
    end else begin
        adpt_stream_read = 1'b0;
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
    if ((1'b1 == ap_block_state6_pp0_stage0_iter5)) begin
        ap_ST_iter5_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter5_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (icmp_ln43_fu_104_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1))) begin
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
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 15'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_66;
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
    if (((1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln43_reg_229_pp0_iter4_reg == 1'd0))) begin
        silu_stream_blk_n = silu_stream_full_n;
    end else begin
        silu_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln43_reg_229_pp0_iter4_reg == 1'd0))) begin
        silu_stream_write = 1'b1;
    end else begin
        silu_stream_write = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (real_start == 1'b1))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b0 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
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
            if ((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else if (((~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter5_fsm_state6) & (icmp_ln43_reg_229_pp0_iter4_reg == 1'd1)) | (~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state6_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state5)))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((ap_done_reg == 1'b1) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
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

assign LUT_IDX_fu_179_p3 = ((icmp_ln43_16_fu_174_p2[0:0] == 1'b1) ? 9'd511 : trunc_ln55_1_reg_250);

assign RELU_fu_193_p3 = ((tmp_61_reg_239_pp0_iter4_reg[0:0] == 1'b1) ? 20'd0 : trunc_ln55_fu_190_p1);

assign SILU_TABLE_address0 = zext_ln57_fu_186_p1;

assign add_ln43_fu_110_p2 = (ap_sig_allocacmp_indvar_flatten_load + 15'd1);

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

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_done_reg == 1'b1) | (ap_start_int == 1'b0) | ((icmp_ln43_fu_104_p2 == 1'd0) & (1'b0 == adpt_stream_empty_n)));
end

always @ (*) begin
    ap_block_state6_pp0_stage0_iter5 = ((icmp_ln43_reg_229_pp0_iter4_reg == 1'd0) & (silu_stream_full_n == 1'b0));
end

always @ (*) begin
    ap_condition_134 = (~((ap_done_reg == 1'b1) | (1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state6) & (1'b1 == ap_block_state6_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_ready = internal_ap_ready;

assign icmp_ln43_16_fu_174_p2 = ((tmp_62_reg_255 != 5'd0) ? 1'b1 : 1'b0);

assign icmp_ln43_fu_104_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 15'd20480) ? 1'b1 : 1'b0);

assign select_ln61_fu_154_p3 = ((tmp_61_reg_239[0:0] == 1'b1) ? tmp_s_fu_144_p4 : tmp_reg_245);

assign shl_ln_fu_204_p3 = {{DIFF_reg_270}, {4'd0}};

assign silu_stream_din = (zext_ln58_fu_200_p1 - zext_ln59_fu_211_p1);

assign start_out = real_start;

assign sub_ln61_fu_139_p2 = (21'd0 - adpt_stream_read_reg_233);

assign tmp_s_fu_144_p4 = {{sub_ln61_fu_139_p2[20:7]}};

assign trunc_ln55_1_fu_160_p1 = select_ln61_fu_154_p3[8:0];

assign trunc_ln55_fu_190_p1 = adpt_stream_read_reg_233_pp0_iter4_reg[19:0];

assign zext_ln57_fu_186_p1 = LUT_IDX_reg_260;

assign zext_ln58_fu_200_p1 = RELU_fu_193_p3;

assign zext_ln59_fu_211_p1 = shl_ln_fu_204_p3;

endmodule //SILU_GELU_do_silu_parallel
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1 ns / 1 ps

  (* use_dsp = "yes" *)  module SILU_GELU_mul_21s_21s_42_4_1(clk,ce,reset,din0, din1, dout);
parameter ID = 1;
parameter NUM_STAGE = 0;
parameter din0_WIDTH = 14;
parameter din1_WIDTH = 12;
parameter dout_WIDTH = 26;

input clk;
input ce;
input reset;

input [din0_WIDTH - 1 : 0] din0; 
input [din1_WIDTH - 1 : 0] din1; 
output [dout_WIDTH - 1 : 0] dout;

wire signed [dout_WIDTH - 1 : 0] tmp_product;


reg signed [dout_WIDTH - 1 : 0] buff0;


reg [din0_WIDTH - 1 :0] din0_reg;
reg [din1_WIDTH - 1 :0] din1_reg;


reg signed [dout_WIDTH - 1 : 0] buff1;












assign tmp_product = $signed(din0_reg) * $signed(din1_reg);




always @(posedge clk)
begin
    if (ce) begin
        buff0 <= tmp_product;

        din0_reg <= din0;
        din1_reg <= din1;


        buff1 <= buff0;




    end
end






assign dout = buff1;




endmodule
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module SILU_GELU_start_for_do_adapt_1_U0
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
    SILU_GELU_start_for_do_adapt_1_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_start_for_do_adapt_1_U0_ShiftReg (
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


module SILU_GELU_start_for_do_adapt_1_U0_ShiftReg
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

module SILU_GELU_do_adapt (
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
        g_stream_dout,
        g_stream_num_data_valid,
        g_stream_fifo_cap,
        g_stream_empty_n,
        g_stream_read,
        adpt_stream_din,
        adpt_stream_num_data_valid,
        adpt_stream_fifo_cap,
        adpt_stream_full_n,
        adpt_stream_write
);

parameter    ap_ST_fsm_state1 = 2'd1;
parameter    ap_ST_fsm_state2 = 2'd2;

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
input  [167:0] g_stream_dout;
input  [4:0] g_stream_num_data_valid;
input  [4:0] g_stream_fifo_cap;
input   g_stream_empty_n;
output   g_stream_read;
output  [20:0] adpt_stream_din;
input  [2:0] adpt_stream_num_data_valid;
input  [2:0] adpt_stream_fifo_cap;
input   adpt_stream_full_n;
output   adpt_stream_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg g_stream_read;
reg adpt_stream_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [1:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
wire    grp_unpk_fu_16_ap_start;
wire    grp_unpk_fu_16_ap_done;
wire    grp_unpk_fu_16_ap_idle;
wire    grp_unpk_fu_16_ap_ready;
wire    grp_unpk_fu_16_g_stream_read;
wire   [20:0] grp_unpk_fu_16_adpt_stream_din;
wire    grp_unpk_fu_16_adpt_stream_write;
reg    grp_unpk_fu_16_ap_start_reg;
reg    ap_block_state1_ignore_call2;
wire    ap_CS_fsm_state2;
reg   [1:0] ap_NS_fsm;
reg    ap_block_state1;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 start_once_reg = 1'b0;
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 2'd1;
//#0 grp_unpk_fu_16_ap_start_reg = 1'b0;
end

SILU_GELU_unpk grp_unpk_fu_16(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_unpk_fu_16_ap_start),
    .ap_done(grp_unpk_fu_16_ap_done),
    .ap_idle(grp_unpk_fu_16_ap_idle),
    .ap_ready(grp_unpk_fu_16_ap_ready),
    .g_stream_dout(g_stream_dout),
    .g_stream_num_data_valid(5'd0),
    .g_stream_fifo_cap(5'd0),
    .g_stream_empty_n(g_stream_empty_n),
    .g_stream_read(grp_unpk_fu_16_g_stream_read),
    .adpt_stream_din(grp_unpk_fu_16_adpt_stream_din),
    .adpt_stream_num_data_valid(3'd0),
    .adpt_stream_fifo_cap(3'd0),
    .adpt_stream_full_n(adpt_stream_full_n),
    .adpt_stream_write(grp_unpk_fu_16_adpt_stream_write)
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
        end else if (((grp_unpk_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_unpk_fu_16_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_state1_ignore_call2) & (1'b1 == ap_CS_fsm_state1))) begin
            grp_unpk_fu_16_ap_start_reg <= 1'b1;
        end else if ((grp_unpk_fu_16_ap_ready == 1'b1)) begin
            grp_unpk_fu_16_ap_start_reg <= 1'b0;
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
    if ((1'b1 == ap_CS_fsm_state2)) begin
        adpt_stream_write = grp_unpk_fu_16_adpt_stream_write;
    end else begin
        adpt_stream_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((grp_unpk_fu_16_ap_done == 1'b0)) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if (((grp_unpk_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
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
    if ((1'b1 == ap_CS_fsm_state2)) begin
        g_stream_read = grp_unpk_fu_16_g_stream_read;
    end else begin
        g_stream_read = 1'b0;
    end
end

always @ (*) begin
    if (((grp_unpk_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((start_full_n == 1'b0) & (start_once_reg == 1'b0))) begin
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
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((grp_unpk_fu_16_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign adpt_stream_din = grp_unpk_fu_16_adpt_stream_din;

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1));
end

always @ (*) begin
    ap_block_state1_ignore_call2 = ((real_start == 1'b0) | (ap_done_reg == 1'b1));
end

assign ap_ready = internal_ap_ready;

assign grp_unpk_fu_16_ap_start = grp_unpk_fu_16_ap_start_reg;

assign start_out = real_start;

endmodule //SILU_GELU_do_adapt
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module SILU_GELU_vit_gelu_to_common_GELU_TABLE_S_ROM_AUTO_1R (
    address0, ce0, q0, 
    address1, ce1, q1, 
    address2, ce2, q2, 
    address3, ce3, q3, 
    address4, ce4, q4, 
    address5, ce5, q5, 
    address6, ce6, q6, 
    address7, ce7, q7, 
    reset, clk);

parameter DataWidth = 4;
parameter AddressWidth = 5;
parameter AddressRange = 32;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;
 
input[AddressWidth-1:0] address1;
input ce1;
output reg[DataWidth-1:0] q1;
 
input[AddressWidth-1:0] address2;
input ce2;
output reg[DataWidth-1:0] q2;
 
input[AddressWidth-1:0] address3;
input ce3;
output reg[DataWidth-1:0] q3;
 
input[AddressWidth-1:0] address4;
input ce4;
output reg[DataWidth-1:0] q4;
 
input[AddressWidth-1:0] address5;
input ce5;
output reg[DataWidth-1:0] q5;
 
input[AddressWidth-1:0] address6;
input ce6;
output reg[DataWidth-1:0] q6;
 
input[AddressWidth-1:0] address7;
input ce7;
output reg[DataWidth-1:0] q7;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];
 
reg [DataWidth-1:0] rom1[0:AddressRange-1];
 
reg [DataWidth-1:0] rom2[0:AddressRange-1];
 
reg [DataWidth-1:0] rom3[0:AddressRange-1];


initial begin
     
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_S_ROM_AUTO_1R.dat", rom0); 
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_S_ROM_AUTO_1R.dat", rom1); 
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_S_ROM_AUTO_1R.dat", rom2); 
    $readmemh("/data/home/songqiangxu/project/VLM/SPINAL/src/main/verilog/SILU_GELU/SILU_GELU_vit_gelu_to_common_GELU_TABLE_S_ROM_AUTO_1R.dat", rom3);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
    end
end
  
always @(posedge clk) 
begin 
    if (ce1) 
    begin
        q1 <= rom0[address1];
    end
end
  
always @(posedge clk) 
begin 
    if (ce2) 
    begin
        q2 <= rom1[address2];
    end
end
  
always @(posedge clk) 
begin 
    if (ce3) 
    begin
        q3 <= rom1[address3];
    end
end
  
always @(posedge clk) 
begin 
    if (ce4) 
    begin
        q4 <= rom2[address4];
    end
end
  
always @(posedge clk) 
begin 
    if (ce5) 
    begin
        q5 <= rom2[address5];
    end
end
  
always @(posedge clk) 
begin 
    if (ce6) 
    begin
        q6 <= rom3[address6];
    end
end
  
always @(posedge clk) 
begin 
    if (ce7) 
    begin
        q7 <= rom3[address7];
    end
end


endmodule

// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module SILU_GELU_start_for_shared_mlp_quantize4_U0
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
    SILU_GELU_start_for_shared_mlp_quantize4_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_SILU_GELU_start_for_shared_mlp_quantize4_U0_ShiftReg (
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


module SILU_GELU_start_for_shared_mlp_quantize4_U0_ShiftReg
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

endmodule