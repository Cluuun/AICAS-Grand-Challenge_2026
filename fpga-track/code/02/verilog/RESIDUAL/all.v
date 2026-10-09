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

module RESIDUAL_regslice_both
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

module RESIDUAL_regslice_both_w1
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

module RESIDUAL_mb_apply_llm_delta (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        x_stream_TVALID,
        res_i_stream_TVALID,
        y_stream_TREADY,
        x_stream_TDATA,
        x_stream_TREADY,
        res_i_stream_TDATA,
        res_i_stream_TREADY,
        y_stream_TDATA,
        y_stream_TVALID
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
input   x_stream_TVALID;
input   res_i_stream_TVALID;
input   y_stream_TREADY;
input  [199:0] x_stream_TDATA;
output   x_stream_TREADY;
input  [199:0] res_i_stream_TDATA;
output   res_i_stream_TREADY;
output  [199:0] y_stream_TDATA;
output   y_stream_TVALID;

reg ap_idle;
reg x_stream_TREADY;
reg res_i_stream_TREADY;
reg y_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire   [0:0] icmp_ln843_fu_103_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
reg   [0:0] icmp_ln843_reg_320;
reg   [0:0] icmp_ln843_reg_320_pp0_iter1_reg;
reg    ap_block_state3_pp0_stage0_iter2;
reg    ap_block_state3_io;
wire    ap_CS_iter2_fsm_state3;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    x_stream_TDATA_blk_n;
reg    res_i_stream_TDATA_blk_n;
reg    y_stream_TDATA_blk_n;
wire   [24:0] x_val_fu_115_p1;
reg   [24:0] x_val_reg_324;
reg   [24:0] x_val_8_reg_329;
reg   [24:0] x_val_9_reg_334;
reg   [24:0] x_val_10_reg_339;
reg   [24:0] x_val_1_reg_344;
reg   [24:0] x_val_2_reg_349;
reg   [24:0] x_val_3_reg_354;
reg   [24:0] x_val_4_reg_359;
wire   [24:0] d_val_fu_189_p1;
reg   [24:0] d_val_reg_364;
reg   [24:0] d_val_8_reg_369;
reg   [24:0] d_val_9_reg_374;
reg   [24:0] d_val_10_reg_379;
reg   [24:0] d_val_1_reg_384;
reg   [24:0] d_val_2_reg_389;
reg   [24:0] d_val_3_reg_394;
reg   [24:0] d_val_4_reg_399;
wire   [24:0] add_ln804_fu_268_p2;
reg   [24:0] add_ln804_reg_404;
wire   [24:0] add_ln804_1_fu_272_p2;
reg   [24:0] add_ln804_1_reg_409;
wire   [24:0] add_ln804_2_fu_276_p2;
reg   [24:0] add_ln804_2_reg_414;
wire   [24:0] add_ln804_3_fu_280_p2;
reg   [24:0] add_ln804_3_reg_419;
wire   [24:0] add_ln804_4_fu_284_p2;
reg   [24:0] add_ln804_4_reg_424;
wire   [24:0] add_ln804_5_fu_288_p2;
reg   [24:0] add_ln804_5_reg_429;
wire   [24:0] add_ln804_6_fu_292_p2;
reg   [24:0] add_ln804_6_reg_434;
wire   [24:0] add_ln804_7_fu_296_p2;
reg   [24:0] add_ln804_7_reg_439;
reg   [9:0] indvar_flatten_fu_72;
wire   [9:0] add_ln843_fu_109_p2;
wire    ap_loop_init;
reg   [9:0] ap_sig_allocacmp_indvar_flatten_load;
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
reg    ap_condition_80;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 indvar_flatten_fu_72 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

RESIDUAL_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_80)) begin
        if ((icmp_ln843_fu_103_p2 == 1'd0)) begin
            indvar_flatten_fu_72 <= add_ln843_fu_109_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_72 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        add_ln804_1_reg_409 <= add_ln804_1_fu_272_p2;
        add_ln804_2_reg_414 <= add_ln804_2_fu_276_p2;
        add_ln804_3_reg_419 <= add_ln804_3_fu_280_p2;
        add_ln804_4_reg_424 <= add_ln804_4_fu_284_p2;
        add_ln804_5_reg_429 <= add_ln804_5_fu_288_p2;
        add_ln804_6_reg_434 <= add_ln804_6_fu_292_p2;
        add_ln804_7_reg_439 <= add_ln804_7_fu_296_p2;
        add_ln804_reg_404 <= add_ln804_fu_268_p2;
        icmp_ln843_reg_320_pp0_iter1_reg <= icmp_ln843_reg_320;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        d_val_10_reg_379 <= {{res_i_stream_TDATA[99:75]}};
        d_val_1_reg_384 <= {{res_i_stream_TDATA[124:100]}};
        d_val_2_reg_389 <= {{res_i_stream_TDATA[149:125]}};
        d_val_3_reg_394 <= {{res_i_stream_TDATA[174:150]}};
        d_val_4_reg_399 <= {{res_i_stream_TDATA[199:175]}};
        d_val_8_reg_369 <= {{res_i_stream_TDATA[49:25]}};
        d_val_9_reg_374 <= {{res_i_stream_TDATA[74:50]}};
        d_val_reg_364 <= d_val_fu_189_p1;
        icmp_ln843_reg_320 <= icmp_ln843_fu_103_p2;
        x_val_10_reg_339 <= {{x_stream_TDATA[99:75]}};
        x_val_1_reg_344 <= {{x_stream_TDATA[124:100]}};
        x_val_2_reg_349 <= {{x_stream_TDATA[149:125]}};
        x_val_3_reg_354 <= {{x_stream_TDATA[174:150]}};
        x_val_4_reg_359 <= {{x_stream_TDATA[199:175]}};
        x_val_8_reg_329 <= {{x_stream_TDATA[49:25]}};
        x_val_9_reg_334 <= {{x_stream_TDATA[74:50]}};
        x_val_reg_324 <= x_val_fu_115_p1;
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
    if (((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) begin
        ap_ST_iter2_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln843_fu_103_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 10'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_72;
    end
end

always @ (*) begin
    if (((icmp_ln843_fu_103_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        res_i_stream_TDATA_blk_n = res_i_stream_TVALID;
    end else begin
        res_i_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln843_fu_103_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        res_i_stream_TREADY = 1'b1;
    end else begin
        res_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln843_fu_103_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        x_stream_TDATA_blk_n = x_stream_TVALID;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln843_fu_103_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        x_stream_TREADY = 1'b1;
    end else begin
        x_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln843_reg_320_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        y_stream_TDATA_blk_n = y_stream_TREADY;
    end else begin
        y_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (icmp_ln843_reg_320_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        y_stream_TVALID = 1'b1;
    end else begin
        y_stream_TVALID = 1'b0;
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
            if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if (((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (icmp_ln843_reg_320_pp0_iter1_reg == 1'd1) & (1'b1 == ap_CS_iter2_fsm_state3)) | (~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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

assign add_ln804_1_fu_272_p2 = (x_val_8_reg_329 + d_val_8_reg_369);

assign add_ln804_2_fu_276_p2 = (x_val_9_reg_334 + d_val_9_reg_374);

assign add_ln804_3_fu_280_p2 = (x_val_10_reg_339 + d_val_10_reg_379);

assign add_ln804_4_fu_284_p2 = (x_val_1_reg_344 + d_val_1_reg_384);

assign add_ln804_5_fu_288_p2 = (x_val_2_reg_349 + d_val_2_reg_389);

assign add_ln804_6_fu_292_p2 = (x_val_3_reg_354 + d_val_3_reg_394);

assign add_ln804_7_fu_296_p2 = (x_val_4_reg_359 + d_val_4_reg_399);

assign add_ln804_fu_268_p2 = (d_val_reg_364 + x_val_reg_324);

assign add_ln843_fu_109_p2 = (ap_sig_allocacmp_indvar_flatten_load + 10'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((res_i_stream_TVALID == 1'b0) & (icmp_ln843_fu_103_p2 == 1'd0)) | ((icmp_ln843_fu_103_p2 == 1'd0) & (x_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state3_io = ((icmp_ln843_reg_320_pp0_iter1_reg == 1'd0) & (y_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state3_pp0_stage0_iter2 = ((icmp_ln843_reg_320_pp0_iter1_reg == 1'd0) & (y_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_80 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign d_val_fu_189_p1 = res_i_stream_TDATA[24:0];

assign icmp_ln843_fu_103_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 10'd960) ? 1'b1 : 1'b0);

assign x_val_fu_115_p1 = x_stream_TDATA[24:0];

assign y_stream_TDATA = {{{{{{{{add_ln804_7_reg_439}, {add_ln804_6_reg_434}}, {add_ln804_5_reg_429}}, {add_ln804_4_reg_424}}, {add_ln804_3_reg_419}}, {add_ln804_2_reg_414}}, {add_ln804_1_reg_409}}, {add_ln804_reg_404}};

endmodule //RESIDUAL_mb_apply_llm_delta
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RESIDUAL_mb_apply_vit_delta (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        x_stream_TVALID,
        res_i_stream_TVALID,
        y_stream_TREADY,
        x_stream_TDATA,
        x_stream_TREADY,
        res_i_stream_TDATA,
        res_i_stream_TREADY,
        y_stream_TDATA,
        y_stream_TVALID
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
input   x_stream_TVALID;
input   res_i_stream_TVALID;
input   y_stream_TREADY;
input  [199:0] x_stream_TDATA;
output   x_stream_TREADY;
input  [199:0] res_i_stream_TDATA;
output   res_i_stream_TREADY;
output  [199:0] y_stream_TDATA;
output   y_stream_TVALID;

reg ap_idle;
reg x_stream_TREADY;
reg res_i_stream_TREADY;
reg y_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire   [0:0] icmp_ln859_fu_103_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
reg   [0:0] icmp_ln859_reg_352;
reg   [0:0] icmp_ln859_reg_352_pp0_iter1_reg;
reg    ap_block_state3_pp0_stage0_iter2;
reg    ap_block_state3_io;
wire    ap_CS_iter2_fsm_state3;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    x_stream_TDATA_blk_n;
reg    res_i_stream_TDATA_blk_n;
reg    y_stream_TDATA_blk_n;
wire   [22:0] out_15_fu_115_p1;
reg   [22:0] out_15_reg_356;
wire   [22:0] d_val_fu_119_p1;
reg   [22:0] d_val_reg_361;
reg   [22:0] out_16_reg_366;
reg   [22:0] d_val_1_reg_371;
reg   [22:0] out_17_reg_376;
reg   [22:0] d_val_2_reg_381;
reg   [22:0] out_18_reg_386;
reg   [22:0] d_val_3_reg_391;
reg   [22:0] out_19_reg_396;
reg   [22:0] d_val_4_reg_401;
reg   [22:0] out_20_reg_406;
reg   [22:0] d_val_5_reg_411;
reg   [22:0] out_21_reg_416;
reg   [22:0] d_val_6_reg_421;
reg   [22:0] out_reg_426;
reg   [22:0] d_val_7_reg_431;
wire   [22:0] value_fu_268_p2;
reg   [22:0] value_reg_436;
wire   [22:0] value_1_fu_272_p2;
reg   [22:0] value_1_reg_441;
wire   [22:0] value_2_fu_276_p2;
reg   [22:0] value_2_reg_446;
wire   [22:0] value_3_fu_280_p2;
reg   [22:0] value_3_reg_451;
wire   [22:0] value_4_fu_284_p2;
reg   [22:0] value_4_reg_456;
wire   [22:0] value_5_fu_288_p2;
reg   [22:0] value_5_reg_461;
wire   [22:0] value_6_fu_292_p2;
reg   [22:0] value_6_reg_466;
wire   [22:0] value_7_fu_296_p2;
reg   [22:0] value_7_reg_471;
reg   [16:0] indvar_flatten10_fu_72;
wire   [16:0] add_ln859_fu_109_p2;
wire    ap_loop_init;
reg   [16:0] ap_sig_allocacmp_indvar_flatten10_load;
wire  signed [24:0] sext_ln865_5_fu_318_p1;
wire  signed [24:0] sext_ln865_4_fu_315_p1;
wire  signed [24:0] sext_ln865_3_fu_312_p1;
wire  signed [24:0] sext_ln865_2_fu_309_p1;
wire  signed [24:0] sext_ln865_1_fu_306_p1;
wire  signed [24:0] sext_ln865_fu_303_p1;
wire  signed [24:0] sext_ln141_fu_300_p1;
wire   [197:0] tmp_fu_321_p9;
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
reg    ap_condition_80;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 indvar_flatten10_fu_72 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

RESIDUAL_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_80)) begin
        if ((icmp_ln859_fu_103_p2 == 1'd0)) begin
            indvar_flatten10_fu_72 <= add_ln859_fu_109_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten10_fu_72 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        d_val_1_reg_371 <= {{res_i_stream_TDATA[47:25]}};
        d_val_2_reg_381 <= {{res_i_stream_TDATA[72:50]}};
        d_val_3_reg_391 <= {{res_i_stream_TDATA[97:75]}};
        d_val_4_reg_401 <= {{res_i_stream_TDATA[122:100]}};
        d_val_5_reg_411 <= {{res_i_stream_TDATA[147:125]}};
        d_val_6_reg_421 <= {{res_i_stream_TDATA[172:150]}};
        d_val_7_reg_431 <= {{res_i_stream_TDATA[197:175]}};
        d_val_reg_361 <= d_val_fu_119_p1;
        icmp_ln859_reg_352 <= icmp_ln859_fu_103_p2;
        out_15_reg_356 <= out_15_fu_115_p1;
        out_16_reg_366 <= {{x_stream_TDATA[47:25]}};
        out_17_reg_376 <= {{x_stream_TDATA[72:50]}};
        out_18_reg_386 <= {{x_stream_TDATA[97:75]}};
        out_19_reg_396 <= {{x_stream_TDATA[122:100]}};
        out_20_reg_406 <= {{x_stream_TDATA[147:125]}};
        out_21_reg_416 <= {{x_stream_TDATA[172:150]}};
        out_reg_426 <= {{x_stream_TDATA[197:175]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        icmp_ln859_reg_352_pp0_iter1_reg <= icmp_ln859_reg_352;
        value_1_reg_441 <= value_1_fu_272_p2;
        value_2_reg_446 <= value_2_fu_276_p2;
        value_3_reg_451 <= value_3_fu_280_p2;
        value_4_reg_456 <= value_4_fu_284_p2;
        value_5_reg_461 <= value_5_fu_288_p2;
        value_6_reg_466 <= value_6_fu_292_p2;
        value_7_reg_471 <= value_7_fu_296_p2;
        value_reg_436 <= value_fu_268_p2;
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
    if (((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) begin
        ap_ST_iter2_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln859_fu_103_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten10_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten10_load = indvar_flatten10_fu_72;
    end
end

always @ (*) begin
    if (((icmp_ln859_fu_103_p2 == 1'd0) & (ap_start_int == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        res_i_stream_TDATA_blk_n = res_i_stream_TVALID;
    end else begin
        res_i_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln859_fu_103_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        res_i_stream_TREADY = 1'b1;
    end else begin
        res_i_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln859_fu_103_p2 == 1'd0) & (ap_start_int == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        x_stream_TDATA_blk_n = x_stream_TVALID;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln859_fu_103_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        x_stream_TREADY = 1'b1;
    end else begin
        x_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln859_reg_352_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        y_stream_TDATA_blk_n = y_stream_TREADY;
    end else begin
        y_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (icmp_ln859_reg_352_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        y_stream_TVALID = 1'b1;
    end else begin
        y_stream_TVALID = 1'b0;
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
            if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if (((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (icmp_ln859_reg_352_pp0_iter1_reg == 1'd1) & (1'b1 == ap_CS_iter2_fsm_state3)) | (~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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

assign add_ln859_fu_109_p2 = (ap_sig_allocacmp_indvar_flatten10_load + 17'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((res_i_stream_TVALID == 1'b0) & (icmp_ln859_fu_103_p2 == 1'd0)) | ((icmp_ln859_fu_103_p2 == 1'd0) & (x_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state3_io = ((icmp_ln859_reg_352_pp0_iter1_reg == 1'd0) & (y_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state3_pp0_stage0_iter2 = ((icmp_ln859_reg_352_pp0_iter1_reg == 1'd0) & (y_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_80 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign d_val_fu_119_p1 = res_i_stream_TDATA[22:0];

assign icmp_ln859_fu_103_p2 = ((ap_sig_allocacmp_indvar_flatten10_load == 17'd98304) ? 1'b1 : 1'b0);

assign out_15_fu_115_p1 = x_stream_TDATA[22:0];

assign sext_ln141_fu_300_p1 = $signed(value_reg_436);

assign sext_ln865_1_fu_306_p1 = $signed(value_2_reg_446);

assign sext_ln865_2_fu_309_p1 = $signed(value_3_reg_451);

assign sext_ln865_3_fu_312_p1 = $signed(value_4_reg_456);

assign sext_ln865_4_fu_315_p1 = $signed(value_5_reg_461);

assign sext_ln865_5_fu_318_p1 = $signed(value_6_reg_466);

assign sext_ln865_fu_303_p1 = $signed(value_1_reg_441);

assign tmp_fu_321_p9 = {{{{{{{{value_7_reg_471}, {sext_ln865_5_fu_318_p1}}, {sext_ln865_4_fu_315_p1}}, {sext_ln865_3_fu_312_p1}}, {sext_ln865_2_fu_309_p1}}, {sext_ln865_1_fu_306_p1}}, {sext_ln865_fu_303_p1}}, {sext_ln141_fu_300_p1}};

assign value_1_fu_272_p2 = (d_val_1_reg_371 + out_16_reg_366);

assign value_2_fu_276_p2 = (d_val_2_reg_381 + out_17_reg_376);

assign value_3_fu_280_p2 = (d_val_3_reg_391 + out_18_reg_386);

assign value_4_fu_284_p2 = (d_val_4_reg_401 + out_19_reg_396);

assign value_5_fu_288_p2 = (d_val_5_reg_411 + out_20_reg_406);

assign value_6_fu_292_p2 = (d_val_6_reg_421 + out_21_reg_416);

assign value_7_fu_296_p2 = (d_val_7_reg_431 + out_reg_426);

assign value_fu_268_p2 = (d_val_reg_361 + out_15_reg_356);

assign y_stream_TDATA = $signed(tmp_fu_321_p9);

endmodule //RESIDUAL_mb_apply_vit_delta
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        x_stream_TVALID,
        res_o_stream_TREADY,
        x_stream_TDATA,
        x_stream_TREADY,
        res_o_stream_TDATA,
        res_o_stream_TVALID
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
input   x_stream_TVALID;
input   res_o_stream_TREADY;
input  [199:0] x_stream_TDATA;
output   x_stream_TREADY;
output  [199:0] res_o_stream_TDATA;
output   res_o_stream_TVALID;

reg ap_idle;
reg x_stream_TREADY;
reg res_o_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln828_fu_63_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln828_reg_87;
wire   [0:0] icmp_ln828_reg_87_pp0_iter0_reg;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    x_stream_TDATA_blk_n;
reg    res_o_stream_TDATA_blk_n;
reg   [199:0] x_stream_read_reg_91;
reg   [9:0] indvar_flatten19_fu_38;
wire   [9:0] add_ln828_fu_69_p2;
wire    ap_loop_init;
reg   [9:0] ap_sig_allocacmp_indvar_flatten19_load;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
wire    ap_start_int;
reg    ap_condition_63;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 indvar_flatten19_fu_38 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

RESIDUAL_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_loop_exit_ready == 1'b0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_63)) begin
        if ((icmp_ln828_fu_63_p2 == 1'd0)) begin
            indvar_flatten19_fu_38 <= add_ln828_fu_69_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten19_fu_38 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln828_reg_87 <= icmp_ln828_fu_63_p2;
        x_stream_read_reg_91 <= x_stream_TDATA;
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
    if (((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1))) begin
        ap_ST_iter1_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln828_fu_63_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_sig_allocacmp_indvar_flatten19_load = 10'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten19_load = indvar_flatten19_fu_38;
    end
end

always @ (*) begin
    if (((icmp_ln828_reg_87 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        res_o_stream_TDATA_blk_n = res_o_stream_TREADY;
    end else begin
        res_o_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln828_reg_87 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        res_o_stream_TVALID = 1'b1;
    end else begin
        res_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln828_fu_63_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        x_stream_TDATA_blk_n = x_stream_TVALID;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln828_fu_63_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        x_stream_TREADY = 1'b1;
    end else begin
        x_stream_TREADY = 1'b0;
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
            if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln828_reg_87_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1)))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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

assign add_ln828_fu_69_p2 = (ap_sig_allocacmp_indvar_flatten19_load + 10'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln828_fu_63_p2 == 1'd0) & (x_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln828_reg_87 == 1'd0) & (res_o_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((icmp_ln828_reg_87 == 1'd0) & (res_o_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_63 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln828_fu_63_p2 = ((ap_sig_allocacmp_indvar_flatten19_load == 10'd960) ? 1'b1 : 1'b0);

assign icmp_ln828_reg_87_pp0_iter0_reg = icmp_ln828_reg_87;

assign res_o_stream_TDATA = x_stream_read_reg_91;

endmodule //RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="RESIDUAL_RESIDUAL,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xczu5ev-fbvb900-1L-i,HLS_INPUT_CLOCK=2.500000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=2.429188,HLS_SYN_LAT=-1,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=2204,HLS_SYN_LUT=2239,HLS_VERSION=2023_2}" *)

module RESIDUAL (
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
        x_stream_TDATA,
        x_stream_TVALID,
        x_stream_TREADY,
        res_i_stream_TDATA,
        res_i_stream_TVALID,
        res_i_stream_TREADY,
        res_o_stream_TDATA,
        res_o_stream_TVALID,
        res_o_stream_TREADY,
        y_stream_TDATA,
        y_stream_TVALID,
        y_stream_TREADY
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
input   ap_rst_n;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [0:0] mode;
input  [31:0] l_begin;
input  [31:0] l_close;
input  [199:0] x_stream_TDATA;
input   x_stream_TVALID;
output   x_stream_TREADY;
input  [199:0] res_i_stream_TDATA;
input   res_i_stream_TVALID;
output   res_i_stream_TREADY;
output  [199:0] res_o_stream_TDATA;
output   res_o_stream_TVALID;
input   res_o_stream_TREADY;
output  [199:0] y_stream_TDATA;
output   y_stream_TVALID;
input   y_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [17:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [31:0] grp_fu_148_p2;
reg   [31:0] reg_160;
reg    ap_block_state1;
wire   [0:0] mode_read_read_fu_98_p2;
wire   [0:0] or_ln880_fu_180_p2;
wire   [30:0] trunc_ln910_fu_164_p1;
reg   [30:0] trunc_ln910_reg_363;
reg   [0:0] or_ln880_reg_369;
wire   [0:0] grp_fu_154_p2;
reg   [0:0] empty_13_reg_380;
wire   [0:0] empty_14_fu_186_p2;
reg   [0:0] empty_14_reg_385;
reg   [0:0] empty_reg_397;
wire   [0:0] empty_9_fu_192_p2;
reg   [0:0] empty_9_reg_402;
wire   [31:0] sub_ln885_fu_208_p2;
reg   [31:0] sub_ln885_reg_407;
wire    ap_CS_fsm_state2;
wire   [31:0] sub_ln885_1_fu_224_p2;
reg   [31:0] sub_ln885_1_reg_413;
wire   [32:0] tmp_1_fu_251_p3;
reg   [32:0] tmp_1_reg_419;
wire    ap_CS_fsm_state3;
wire   [31:0] sub_ln901_fu_283_p2;
reg   [31:0] sub_ln901_reg_427;
wire    ap_CS_fsm_state11;
wire   [31:0] sub_ln901_1_fu_299_p2;
reg   [31:0] sub_ln901_1_reg_433;
wire   [32:0] tmp_fu_321_p3;
reg   [32:0] tmp_reg_439;
wire    ap_CS_fsm_state12;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_done;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_idle;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_ready;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TREADY;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_x_stream_TREADY;
wire   [199:0] grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TDATA;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TVALID;
wire    grp_mb_apply_llm_delta_fu_112_ap_start;
wire    grp_mb_apply_llm_delta_fu_112_ap_done;
wire    grp_mb_apply_llm_delta_fu_112_ap_idle;
wire    grp_mb_apply_llm_delta_fu_112_ap_ready;
wire    grp_mb_apply_llm_delta_fu_112_y_stream_TREADY;
wire    grp_mb_apply_llm_delta_fu_112_x_stream_TREADY;
wire    grp_mb_apply_llm_delta_fu_112_res_i_stream_TREADY;
wire   [199:0] grp_mb_apply_llm_delta_fu_112_y_stream_TDATA;
wire    grp_mb_apply_llm_delta_fu_112_y_stream_TVALID;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_done;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_idle;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_ready;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TREADY;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_x_stream_TREADY;
wire   [199:0] grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TDATA;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TVALID;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_done;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_idle;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_ready;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TREADY;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_x_stream_TREADY;
wire   [199:0] grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TDATA;
wire    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TVALID;
wire    grp_mb_apply_vit_delta_fu_138_ap_start;
wire    grp_mb_apply_vit_delta_fu_138_ap_done;
wire    grp_mb_apply_vit_delta_fu_138_ap_idle;
wire    grp_mb_apply_vit_delta_fu_138_ap_ready;
wire    grp_mb_apply_vit_delta_fu_138_y_stream_TREADY;
wire    grp_mb_apply_vit_delta_fu_138_x_stream_TREADY;
wire    grp_mb_apply_vit_delta_fu_138_res_i_stream_TREADY;
wire   [199:0] grp_mb_apply_vit_delta_fu_138_y_stream_TDATA;
wire    grp_mb_apply_vit_delta_fu_138_y_stream_TVALID;
reg    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start_reg;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state6;
reg    grp_mb_apply_llm_delta_fu_112_ap_start_reg;
wire    ap_CS_fsm_state7;
wire    ap_CS_fsm_state8;
reg    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start_reg;
wire    ap_CS_fsm_state9;
wire    ap_CS_fsm_state10;
reg    grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start_reg;
wire    ap_CS_fsm_state14;
wire    ap_CS_fsm_state15;
reg    grp_mb_apply_vit_delta_fu_138_ap_start_reg;
wire    ap_CS_fsm_state16;
wire    ap_CS_fsm_state17;
reg   [32:0] indvar_flatten37_fu_78;
wire   [32:0] add_ln885_1_fu_267_p2;
wire    ap_CS_fsm_state4;
wire   [0:0] icmp_ln885_fu_262_p2;
reg   [32:0] indvar_flatten12_fu_82;
wire   [32:0] add_ln901_1_fu_337_p2;
wire    ap_CS_fsm_state13;
wire   [0:0] icmp_ln901_fu_332_p2;
wire   [0:0] icmp_ln880_fu_168_p2;
wire   [0:0] icmp_ln880_1_fu_174_p2;
wire   [31:0] smax_fu_203_p3;
wire   [30:0] smax1_fu_214_p3;
wire   [31:0] zext_ln885_fu_220_p1;
wire   [0:0] empty_15_fu_235_p2;
wire   [31:0] umax_fu_239_p3;
wire   [31:0] xor_ln885_fu_245_p2;
wire   [31:0] smax2_fu_278_p3;
wire   [30:0] smax3_fu_289_p3;
wire   [31:0] zext_ln901_fu_295_p1;
wire   [0:0] empty_10_fu_305_p2;
wire   [31:0] umax4_fu_309_p3;
wire   [31:0] xor_ln901_fu_315_p2;
wire    ap_CS_fsm_state18;
wire    regslice_both_res_o_stream_U_apdone_blk;
wire    regslice_both_y_stream_U_apdone_blk;
reg    ap_block_state18;
reg   [17:0] ap_NS_fsm;
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
wire    ap_ST_fsm_state12_blk;
wire    ap_ST_fsm_state13_blk;
wire    ap_ST_fsm_state14_blk;
reg    ap_ST_fsm_state15_blk;
wire    ap_ST_fsm_state16_blk;
reg    ap_ST_fsm_state17_blk;
reg    ap_ST_fsm_state18_blk;
wire    regslice_both_x_stream_U_apdone_blk;
wire   [199:0] x_stream_TDATA_int_regslice;
wire    x_stream_TVALID_int_regslice;
reg    x_stream_TREADY_int_regslice;
wire    regslice_both_x_stream_U_ack_in;
wire    regslice_both_res_i_stream_U_apdone_blk;
wire   [199:0] res_i_stream_TDATA_int_regslice;
wire    res_i_stream_TVALID_int_regslice;
reg    res_i_stream_TREADY_int_regslice;
wire    regslice_both_res_i_stream_U_ack_in;
reg   [199:0] res_o_stream_TDATA_int_regslice;
reg    res_o_stream_TVALID_int_regslice;
wire    res_o_stream_TREADY_int_regslice;
wire    regslice_both_res_o_stream_U_vld_out;
reg   [199:0] y_stream_TDATA_int_regslice;
reg    y_stream_TVALID_int_regslice;
wire    y_stream_TREADY_int_regslice;
wire    regslice_both_y_stream_U_vld_out;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 18'd1;
//#0 grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start_reg = 1'b0;
//#0 grp_mb_apply_llm_delta_fu_112_ap_start_reg = 1'b0;
//#0 grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start_reg = 1'b0;
//#0 grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start_reg = 1'b0;
//#0 grp_mb_apply_vit_delta_fu_138_ap_start_reg = 1'b0;
//#0 indvar_flatten37_fu_78 = 33'd0;
//#0 indvar_flatten12_fu_82 = 33'd0;
end

RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22 grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start),
    .ap_done(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_done),
    .ap_idle(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_idle),
    .ap_ready(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_ready),
    .x_stream_TVALID(x_stream_TVALID_int_regslice),
    .res_o_stream_TREADY(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TREADY),
    .x_stream_TDATA(x_stream_TDATA_int_regslice),
    .x_stream_TREADY(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_x_stream_TREADY),
    .res_o_stream_TDATA(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TDATA),
    .res_o_stream_TVALID(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TVALID)
);

RESIDUAL_mb_apply_llm_delta grp_mb_apply_llm_delta_fu_112(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_mb_apply_llm_delta_fu_112_ap_start),
    .ap_done(grp_mb_apply_llm_delta_fu_112_ap_done),
    .ap_idle(grp_mb_apply_llm_delta_fu_112_ap_idle),
    .ap_ready(grp_mb_apply_llm_delta_fu_112_ap_ready),
    .x_stream_TVALID(x_stream_TVALID_int_regslice),
    .res_i_stream_TVALID(res_i_stream_TVALID_int_regslice),
    .y_stream_TREADY(grp_mb_apply_llm_delta_fu_112_y_stream_TREADY),
    .x_stream_TDATA(x_stream_TDATA_int_regslice),
    .x_stream_TREADY(grp_mb_apply_llm_delta_fu_112_x_stream_TREADY),
    .res_i_stream_TDATA(res_i_stream_TDATA_int_regslice),
    .res_i_stream_TREADY(grp_mb_apply_llm_delta_fu_112_res_i_stream_TREADY),
    .y_stream_TDATA(grp_mb_apply_llm_delta_fu_112_y_stream_TDATA),
    .y_stream_TVALID(grp_mb_apply_llm_delta_fu_112_y_stream_TVALID)
);

RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21 grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start),
    .ap_done(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_done),
    .ap_idle(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_idle),
    .ap_ready(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_ready),
    .x_stream_TVALID(x_stream_TVALID_int_regslice),
    .res_o_stream_TREADY(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TREADY),
    .x_stream_TDATA(x_stream_TDATA_int_regslice),
    .x_stream_TREADY(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_x_stream_TREADY),
    .res_o_stream_TDATA(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TDATA),
    .res_o_stream_TVALID(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TVALID)
);

RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2 grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start),
    .ap_done(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_done),
    .ap_idle(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_idle),
    .ap_ready(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_ready),
    .x_stream_TVALID(x_stream_TVALID_int_regslice),
    .res_o_stream_TREADY(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TREADY),
    .x_stream_TDATA(x_stream_TDATA_int_regslice),
    .x_stream_TREADY(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_x_stream_TREADY),
    .res_o_stream_TDATA(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TDATA),
    .res_o_stream_TVALID(grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TVALID)
);

RESIDUAL_mb_apply_vit_delta grp_mb_apply_vit_delta_fu_138(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_mb_apply_vit_delta_fu_138_ap_start),
    .ap_done(grp_mb_apply_vit_delta_fu_138_ap_done),
    .ap_idle(grp_mb_apply_vit_delta_fu_138_ap_idle),
    .ap_ready(grp_mb_apply_vit_delta_fu_138_ap_ready),
    .x_stream_TVALID(x_stream_TVALID_int_regslice),
    .res_i_stream_TVALID(res_i_stream_TVALID_int_regslice),
    .y_stream_TREADY(grp_mb_apply_vit_delta_fu_138_y_stream_TREADY),
    .x_stream_TDATA(x_stream_TDATA_int_regslice),
    .x_stream_TREADY(grp_mb_apply_vit_delta_fu_138_x_stream_TREADY),
    .res_i_stream_TDATA(res_i_stream_TDATA_int_regslice),
    .res_i_stream_TREADY(grp_mb_apply_vit_delta_fu_138_res_i_stream_TREADY),
    .y_stream_TDATA(grp_mb_apply_vit_delta_fu_138_y_stream_TDATA),
    .y_stream_TVALID(grp_mb_apply_vit_delta_fu_138_y_stream_TVALID)
);

RESIDUAL_regslice_both #(
    .DataWidth( 200 ))
regslice_both_x_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(x_stream_TDATA),
    .vld_in(x_stream_TVALID),
    .ack_in(regslice_both_x_stream_U_ack_in),
    .data_out(x_stream_TDATA_int_regslice),
    .vld_out(x_stream_TVALID_int_regslice),
    .ack_out(x_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_x_stream_U_apdone_blk)
);

RESIDUAL_regslice_both #(
    .DataWidth( 200 ))
regslice_both_res_i_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(res_i_stream_TDATA),
    .vld_in(res_i_stream_TVALID),
    .ack_in(regslice_both_res_i_stream_U_ack_in),
    .data_out(res_i_stream_TDATA_int_regslice),
    .vld_out(res_i_stream_TVALID_int_regslice),
    .ack_out(res_i_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_res_i_stream_U_apdone_blk)
);

RESIDUAL_regslice_both #(
    .DataWidth( 200 ))
regslice_both_res_o_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(res_o_stream_TDATA_int_regslice),
    .vld_in(res_o_stream_TVALID_int_regslice),
    .ack_in(res_o_stream_TREADY_int_regslice),
    .data_out(res_o_stream_TDATA),
    .vld_out(regslice_both_res_o_stream_U_vld_out),
    .ack_out(res_o_stream_TREADY),
    .apdone_blk(regslice_both_res_o_stream_U_apdone_blk)
);

RESIDUAL_regslice_both #(
    .DataWidth( 200 ))
regslice_both_y_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(y_stream_TDATA_int_regslice),
    .vld_in(y_stream_TVALID_int_regslice),
    .ack_in(y_stream_TREADY_int_regslice),
    .data_out(y_stream_TDATA),
    .vld_out(regslice_both_y_stream_U_vld_out),
    .ack_out(y_stream_TREADY),
    .apdone_blk(regslice_both_y_stream_U_apdone_blk)
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
        end else if (((1'b0 == ap_block_state18) & (1'b1 == ap_CS_fsm_state18))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_mb_apply_llm_delta_fu_112_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state7)) begin
            grp_mb_apply_llm_delta_fu_112_ap_start_reg <= 1'b1;
        end else if ((grp_mb_apply_llm_delta_fu_112_ap_ready == 1'b1)) begin
            grp_mb_apply_llm_delta_fu_112_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_mb_apply_vit_delta_fu_138_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state16)) begin
            grp_mb_apply_vit_delta_fu_138_ap_start_reg <= 1'b1;
        end else if ((grp_mb_apply_vit_delta_fu_138_ap_ready == 1'b1)) begin
            grp_mb_apply_vit_delta_fu_138_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state9)) begin
            grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start_reg <= 1'b1;
        end else if ((grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_ready == 1'b1)) begin
            grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state5)) begin
            grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start_reg <= 1'b1;
        end else if ((grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_ready == 1'b1)) begin
            grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state14)) begin
            grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start_reg <= 1'b1;
        end else if ((grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_ready == 1'b1)) begin
            grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((mode_read_read_fu_98_p2 == 1'd1) & (1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        indvar_flatten12_fu_82 <= 33'd0;
    end else if (((icmp_ln901_fu_332_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state13))) begin
        indvar_flatten12_fu_82 <= add_ln901_1_fu_337_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        indvar_flatten37_fu_78 <= 33'd0;
    end else if (((or_ln880_reg_369 == 1'd0) & (icmp_ln885_fu_262_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        indvar_flatten37_fu_78 <= add_ln885_1_fu_267_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        empty_13_reg_380 <= grp_fu_154_p2;
        empty_14_reg_385 <= empty_14_fu_186_p2;
        empty_9_reg_402 <= empty_9_fu_192_p2;
        empty_reg_397 <= grp_fu_154_p2;
        or_ln880_reg_369 <= or_ln880_fu_180_p2;
        trunc_ln910_reg_363 <= trunc_ln910_fu_164_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((((mode_read_read_fu_98_p2 == 1'd1) & (1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1)) | ((or_ln880_fu_180_p2 == 1'd0) & (mode_read_read_fu_98_p2 == 1'd0) & (1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1)))) begin
        reg_160 <= grp_fu_148_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        sub_ln885_1_reg_413 <= sub_ln885_1_fu_224_p2;
        sub_ln885_reg_407 <= sub_ln885_fu_208_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        sub_ln901_1_reg_433 <= sub_ln901_1_fu_299_p2;
        sub_ln901_reg_427 <= sub_ln901_fu_283_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        tmp_1_reg_419[32 : 1] <= tmp_1_fu_251_p3[32 : 1];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state12)) begin
        tmp_reg_439[32 : 1] <= tmp_fu_321_p3[32 : 1];
    end
end

always @ (*) begin
    if ((grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_done == 1'b0)) begin
        ap_ST_fsm_state10_blk = 1'b1;
    end else begin
        ap_ST_fsm_state10_blk = 1'b0;
    end
end

assign ap_ST_fsm_state11_blk = 1'b0;

assign ap_ST_fsm_state12_blk = 1'b0;

assign ap_ST_fsm_state13_blk = 1'b0;

assign ap_ST_fsm_state14_blk = 1'b0;

always @ (*) begin
    if ((grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_done == 1'b0)) begin
        ap_ST_fsm_state15_blk = 1'b1;
    end else begin
        ap_ST_fsm_state15_blk = 1'b0;
    end
end

assign ap_ST_fsm_state16_blk = 1'b0;

always @ (*) begin
    if ((grp_mb_apply_vit_delta_fu_138_ap_done == 1'b0)) begin
        ap_ST_fsm_state17_blk = 1'b1;
    end else begin
        ap_ST_fsm_state17_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state18)) begin
        ap_ST_fsm_state18_blk = 1'b1;
    end else begin
        ap_ST_fsm_state18_blk = 1'b0;
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

always @ (*) begin
    if ((grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_done == 1'b0)) begin
        ap_ST_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_fsm_state6_blk = 1'b0;
    end
end

assign ap_ST_fsm_state7_blk = 1'b0;

always @ (*) begin
    if ((grp_mb_apply_llm_delta_fu_112_ap_done == 1'b0)) begin
        ap_ST_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_fsm_state8_blk = 1'b0;
    end
end

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((1'b0 == ap_block_state18) & (1'b1 == ap_CS_fsm_state18))) begin
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
    if (((1'b0 == ap_block_state18) & (1'b1 == ap_CS_fsm_state18))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state17)) begin
        res_i_stream_TREADY_int_regslice = grp_mb_apply_vit_delta_fu_138_res_i_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state8)) begin
        res_i_stream_TREADY_int_regslice = grp_mb_apply_llm_delta_fu_112_res_i_stream_TREADY;
    end else begin
        res_i_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state15) & (grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TVALID == 1'b1))) begin
        res_o_stream_TDATA_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TDATA;
    end else if (((1'b1 == ap_CS_fsm_state10) & (grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TVALID == 1'b1))) begin
        res_o_stream_TDATA_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TDATA;
    end else if (((1'b1 == ap_CS_fsm_state6) & (grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TVALID == 1'b1))) begin
        res_o_stream_TDATA_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TDATA;
    end else begin
        res_o_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state15)) begin
        res_o_stream_TVALID_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        res_o_stream_TVALID_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        res_o_stream_TVALID_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TVALID;
    end else begin
        res_o_stream_TVALID_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state17)) begin
        x_stream_TREADY_int_regslice = grp_mb_apply_vit_delta_fu_138_x_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state15)) begin
        x_stream_TREADY_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_x_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        x_stream_TREADY_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_x_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state8)) begin
        x_stream_TREADY_int_regslice = grp_mb_apply_llm_delta_fu_112_x_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        x_stream_TREADY_int_regslice = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_x_stream_TREADY;
    end else begin
        x_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state17) & (grp_mb_apply_vit_delta_fu_138_y_stream_TVALID == 1'b1))) begin
        y_stream_TDATA_int_regslice = grp_mb_apply_vit_delta_fu_138_y_stream_TDATA;
    end else if (((1'b1 == ap_CS_fsm_state8) & (grp_mb_apply_llm_delta_fu_112_y_stream_TVALID == 1'b1))) begin
        y_stream_TDATA_int_regslice = grp_mb_apply_llm_delta_fu_112_y_stream_TDATA;
    end else begin
        y_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state17)) begin
        y_stream_TVALID_int_regslice = grp_mb_apply_vit_delta_fu_138_y_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state8)) begin
        y_stream_TVALID_int_regslice = grp_mb_apply_llm_delta_fu_112_y_stream_TVALID;
    end else begin
        y_stream_TVALID_int_regslice = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((or_ln880_fu_180_p2 == 1'd1) & (mode_read_read_fu_98_p2 == 1'd0) & (1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else if (((or_ln880_fu_180_p2 == 1'd0) & (mode_read_read_fu_98_p2 == 1'd0) & (1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else if (((mode_read_read_fu_98_p2 == 1'd1) & (1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state11;
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
            if (((1'b1 == ap_CS_fsm_state4) & ((or_ln880_reg_369 == 1'd1) | (icmp_ln885_fu_262_p2 == 1'd1)))) begin
                ap_NS_fsm = ap_ST_fsm_state18;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_CS_fsm_state6) & (grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            if (((1'b1 == ap_CS_fsm_state8) & (grp_mb_apply_llm_delta_fu_112_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            if (((1'b1 == ap_CS_fsm_state10) & (grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state12;
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            if (((icmp_ln901_fu_332_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state13))) begin
                ap_NS_fsm = ap_ST_fsm_state18;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state15;
        end
        ap_ST_fsm_state15 : begin
            if (((1'b1 == ap_CS_fsm_state15) & (grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state16;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state15;
            end
        end
        ap_ST_fsm_state16 : begin
            ap_NS_fsm = ap_ST_fsm_state17;
        end
        ap_ST_fsm_state17 : begin
            if (((1'b1 == ap_CS_fsm_state17) & (grp_mb_apply_vit_delta_fu_138_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state17;
            end
        end
        ap_ST_fsm_state18 : begin
            if (((1'b0 == ap_block_state18) & (1'b1 == ap_CS_fsm_state18))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state18;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln885_1_fu_267_p2 = (indvar_flatten37_fu_78 + 33'd1);

assign add_ln901_1_fu_337_p2 = (indvar_flatten12_fu_82 + 33'd1);

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
    ap_block_state1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0));
end

always @ (*) begin
    ap_block_state18 = ((regslice_both_y_stream_U_apdone_blk == 1'b1) | (regslice_both_res_o_stream_U_apdone_blk == 1'b1));
end

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign empty_10_fu_305_p2 = ((sub_ln901_reg_427 > sub_ln901_1_reg_433) ? 1'b1 : 1'b0);

assign empty_14_fu_186_p2 = (($signed(l_begin) > $signed(32'd32)) ? 1'b1 : 1'b0);

assign empty_15_fu_235_p2 = ((sub_ln885_reg_407 > sub_ln885_1_reg_413) ? 1'b1 : 1'b0);

assign empty_9_fu_192_p2 = (($signed(l_begin) > $signed(32'd12)) ? 1'b1 : 1'b0);

assign grp_fu_148_p2 = ($signed(l_begin) + $signed(32'd4294967295));

assign grp_fu_154_p2 = (($signed(l_close) > $signed(l_begin)) ? 1'b1 : 1'b0);

assign grp_mb_apply_llm_delta_fu_112_ap_start = grp_mb_apply_llm_delta_fu_112_ap_start_reg;

assign grp_mb_apply_llm_delta_fu_112_y_stream_TREADY = (y_stream_TREADY_int_regslice & ap_CS_fsm_state8);

assign grp_mb_apply_vit_delta_fu_138_ap_start = grp_mb_apply_vit_delta_fu_138_ap_start_reg;

assign grp_mb_apply_vit_delta_fu_138_y_stream_TREADY = (y_stream_TREADY_int_regslice & ap_CS_fsm_state17);

assign grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_ap_start_reg;

assign grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_21_fu_122_res_o_stream_TREADY = (res_o_stream_TREADY_int_regslice & ap_CS_fsm_state10);

assign grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_ap_start_reg;

assign grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22_fu_104_res_o_stream_TREADY = (res_o_stream_TREADY_int_regslice & ap_CS_fsm_state6);

assign grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start = grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_ap_start_reg;

assign grp_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2_fu_130_res_o_stream_TREADY = (res_o_stream_TREADY_int_regslice & ap_CS_fsm_state15);

assign icmp_ln880_1_fu_174_p2 = ((l_close == 32'd33) ? 1'b1 : 1'b0);

assign icmp_ln880_fu_168_p2 = ((l_begin == 32'd32) ? 1'b1 : 1'b0);

assign icmp_ln885_fu_262_p2 = ((indvar_flatten37_fu_78 == tmp_1_reg_419) ? 1'b1 : 1'b0);

assign icmp_ln901_fu_332_p2 = ((indvar_flatten12_fu_82 == tmp_reg_439) ? 1'b1 : 1'b0);

assign mode_read_read_fu_98_p2 = mode;

assign or_ln880_fu_180_p2 = (icmp_ln880_fu_168_p2 | icmp_ln880_1_fu_174_p2);

assign res_i_stream_TREADY = regslice_both_res_i_stream_U_ack_in;

assign res_o_stream_TVALID = regslice_both_res_o_stream_U_vld_out;

assign smax1_fu_214_p3 = ((empty_14_reg_385[0:0] == 1'b1) ? trunc_ln910_reg_363 : 31'd32);

assign smax2_fu_278_p3 = ((empty_reg_397[0:0] == 1'b1) ? l_close : l_begin);

assign smax3_fu_289_p3 = ((empty_9_reg_402[0:0] == 1'b1) ? trunc_ln910_reg_363 : 31'd12);

assign smax_fu_203_p3 = ((empty_13_reg_380[0:0] == 1'b1) ? l_close : l_begin);

assign sub_ln885_1_fu_224_p2 = (reg_160 - zext_ln885_fu_220_p1);

assign sub_ln885_fu_208_p2 = (reg_160 - smax_fu_203_p3);

assign sub_ln901_1_fu_299_p2 = (reg_160 - zext_ln901_fu_295_p1);

assign sub_ln901_fu_283_p2 = (reg_160 - smax2_fu_278_p3);

assign tmp_1_fu_251_p3 = {{xor_ln885_fu_245_p2}, {1'd0}};

assign tmp_fu_321_p3 = {{xor_ln901_fu_315_p2}, {1'd0}};

assign trunc_ln910_fu_164_p1 = l_begin[30:0];

assign umax4_fu_309_p3 = ((empty_10_fu_305_p2[0:0] == 1'b1) ? sub_ln901_reg_427 : sub_ln901_1_reg_433);

assign umax_fu_239_p3 = ((empty_15_fu_235_p2[0:0] == 1'b1) ? sub_ln885_reg_407 : sub_ln885_1_reg_413);

assign x_stream_TREADY = regslice_both_x_stream_U_ack_in;

assign xor_ln885_fu_245_p2 = (umax_fu_239_p3 ^ 32'd4294967295);

assign xor_ln901_fu_315_p2 = (umax4_fu_309_p3 ^ 32'd4294967295);

assign y_stream_TVALID = regslice_both_y_stream_U_vld_out;

assign zext_ln885_fu_220_p1 = smax1_fu_214_p3;

assign zext_ln901_fu_295_p1 = smax3_fu_289_p3;

always @ (posedge ap_clk) begin
    tmp_1_reg_419[0] <= 1'b0;
    tmp_reg_439[0] <= 1'b0;
end

endmodule //RESIDUAL
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        x_stream_TVALID,
        res_o_stream_TREADY,
        x_stream_TDATA,
        x_stream_TREADY,
        res_o_stream_TDATA,
        res_o_stream_TVALID
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
input   x_stream_TVALID;
input   res_o_stream_TREADY;
input  [199:0] x_stream_TDATA;
output   x_stream_TREADY;
output  [199:0] res_o_stream_TDATA;
output   res_o_stream_TVALID;

reg ap_idle;
reg x_stream_TREADY;
reg res_o_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln828_fu_63_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln828_reg_87;
wire   [0:0] icmp_ln828_reg_87_pp0_iter0_reg;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    x_stream_TDATA_blk_n;
reg    res_o_stream_TDATA_blk_n;
reg   [199:0] x_stream_read_reg_91;
reg   [9:0] indvar_flatten26_fu_38;
wire   [9:0] add_ln828_fu_69_p2;
wire    ap_loop_init;
reg   [9:0] ap_sig_allocacmp_indvar_flatten26_load;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
wire    ap_start_int;
reg    ap_condition_63;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 indvar_flatten26_fu_38 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

RESIDUAL_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_loop_exit_ready == 1'b0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_63)) begin
        if ((icmp_ln828_fu_63_p2 == 1'd0)) begin
            indvar_flatten26_fu_38 <= add_ln828_fu_69_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten26_fu_38 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln828_reg_87 <= icmp_ln828_fu_63_p2;
        x_stream_read_reg_91 <= x_stream_TDATA;
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
    if (((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1))) begin
        ap_ST_iter1_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln828_fu_63_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_sig_allocacmp_indvar_flatten26_load = 10'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten26_load = indvar_flatten26_fu_38;
    end
end

always @ (*) begin
    if (((icmp_ln828_reg_87 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        res_o_stream_TDATA_blk_n = res_o_stream_TREADY;
    end else begin
        res_o_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln828_reg_87 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        res_o_stream_TVALID = 1'b1;
    end else begin
        res_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln828_fu_63_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        x_stream_TDATA_blk_n = x_stream_TVALID;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln828_fu_63_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        x_stream_TREADY = 1'b1;
    end else begin
        x_stream_TREADY = 1'b0;
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
            if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln828_reg_87_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1)))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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

assign add_ln828_fu_69_p2 = (ap_sig_allocacmp_indvar_flatten26_load + 10'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln828_fu_63_p2 == 1'd0) & (x_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln828_reg_87 == 1'd0) & (res_o_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((icmp_ln828_reg_87 == 1'd0) & (res_o_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_63 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln828_fu_63_p2 = ((ap_sig_allocacmp_indvar_flatten26_load == 10'd960) ? 1'b1 : 1'b0);

assign icmp_ln828_reg_87_pp0_iter0_reg = icmp_ln828_reg_87;

assign res_o_stream_TDATA = x_stream_read_reg_91;

endmodule //RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_22
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module RESIDUAL_flow_control_loop_pipe_sequential_init(
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

// if no ap_continue port and current module is not RESIDUAL module, 
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

// if no ap_continue port and current module is not RESIDUAL module, ap_done handshakes with ap_start
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

module RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        x_stream_TVALID,
        res_o_stream_TREADY,
        x_stream_TDATA,
        x_stream_TREADY,
        res_o_stream_TDATA,
        res_o_stream_TVALID
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
input   x_stream_TVALID;
input   res_o_stream_TREADY;
input  [199:0] x_stream_TDATA;
output   x_stream_TREADY;
output  [199:0] res_o_stream_TDATA;
output   res_o_stream_TVALID;

reg ap_idle;
reg x_stream_TREADY;
reg res_o_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln828_fu_63_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln828_reg_87;
wire   [0:0] icmp_ln828_reg_87_pp0_iter0_reg;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    x_stream_TDATA_blk_n;
reg    res_o_stream_TDATA_blk_n;
reg   [199:0] x_stream_read_reg_91;
reg   [16:0] indvar_flatten_fu_38;
wire   [16:0] add_ln828_fu_69_p2;
wire    ap_loop_init;
reg   [16:0] ap_sig_allocacmp_indvar_flatten_load;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
wire    ap_start_int;
reg    ap_condition_63;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 indvar_flatten_fu_38 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

RESIDUAL_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_loop_exit_ready == 1'b0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_63)) begin
        if ((icmp_ln828_fu_63_p2 == 1'd0)) begin
            indvar_flatten_fu_38 <= add_ln828_fu_69_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_38 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln828_reg_87 <= icmp_ln828_fu_63_p2;
        x_stream_read_reg_91 <= x_stream_TDATA;
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
    if (((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1))) begin
        ap_ST_iter1_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln828_fu_63_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1))) begin
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_38;
    end
end

always @ (*) begin
    if (((icmp_ln828_reg_87 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        res_o_stream_TDATA_blk_n = res_o_stream_TREADY;
    end else begin
        res_o_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln828_reg_87 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        res_o_stream_TVALID = 1'b1;
    end else begin
        res_o_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln828_fu_63_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        x_stream_TDATA_blk_n = x_stream_TVALID;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln828_fu_63_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        x_stream_TREADY = 1'b1;
    end else begin
        x_stream_TREADY = 1'b0;
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
            if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln828_reg_87_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1)))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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

assign add_ln828_fu_69_p2 = (ap_sig_allocacmp_indvar_flatten_load + 17'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln828_fu_63_p2 == 1'd0) & (x_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln828_reg_87 == 1'd0) & (res_o_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((icmp_ln828_reg_87 == 1'd0) & (res_o_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_63 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln828_fu_63_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 17'd98304) ? 1'b1 : 1'b0);

assign icmp_ln828_reg_87_pp0_iter0_reg = icmp_ln828_reg_87;

assign res_o_stream_TDATA = x_stream_read_reg_91;

endmodule //RESIDUAL_RESIDUAL_Pipeline_VITIS_LOOP_828_1_VITIS_LOOP_829_2
