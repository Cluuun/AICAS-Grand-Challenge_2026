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

module DEMUX_regslice_both
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

module DEMUX_regslice_both_w1
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

module DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TVALID,
        od_fc2_stream_TREADY,
        mlp1_stream_TREADY,
        od_fc2_stream_TDATA,
        od_fc2_stream_TVALID,
        gemm_stream_TDATA,
        gemm_stream_TREADY,
        mlp1_stream_TDATA,
        mlp1_stream_TVALID
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
input   gemm_stream_TVALID;
input   od_fc2_stream_TREADY;
input   mlp1_stream_TREADY;
output  [199:0] od_fc2_stream_TDATA;
output   od_fc2_stream_TVALID;
input  [239:0] gemm_stream_TDATA;
output   gemm_stream_TREADY;
output  [223:0] mlp1_stream_TDATA;
output   mlp1_stream_TVALID;

reg ap_idle;
reg[199:0] od_fc2_stream_TDATA;
reg od_fc2_stream_TVALID;
reg gemm_stream_TREADY;
reg mlp1_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln150_fu_184_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln150_reg_563;
wire   [0:0] icmp_ln150_reg_563_pp0_iter0_reg;
reg   [0:0] icmp_ln153_reg_623;
reg   [0:0] icmp_ln162_reg_627;
reg    ap_predicate_op42_write_state2;
reg    ap_predicate_op61_write_state2;
reg    ap_predicate_op73_write_state2;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    od_fc2_stream_TDATA_blk_n;
reg    gemm_stream_TDATA_blk_n;
reg    mlp1_stream_TDATA_blk_n;
reg   [239:0] gemm_stream_read_reg_567;
reg   [20:0] lshr_ln_reg_588;
reg   [20:0] local_o_16_reg_593;
reg   [20:0] local_o_1_reg_598;
reg   [20:0] local_o_2_reg_603;
reg   [20:0] local_o_3_reg_608;
reg   [20:0] local_o_4_reg_613;
reg   [20:0] lshr_ln152_6_reg_618;
wire   [0:0] icmp_ln153_fu_266_p2;
wire   [0:0] icmp_ln162_fu_272_p2;
reg   [12:0] iter_fu_152;
wire   [12:0] iter_5_fu_190_p2;
wire    ap_loop_init;
reg   [12:0] ap_sig_allocacmp_iter_4;
wire   [199:0] p_s_fu_355_p9;
wire  signed [199:0] sext_ln161_1_fu_551_p1;
wire   [24:0] local_d_7_fu_346_p4;
wire   [24:0] local_d_6_fu_337_p4;
wire   [24:0] local_d_5_fu_328_p4;
wire   [24:0] local_d_4_fu_319_p4;
wire   [24:0] local_d_3_fu_310_p4;
wire   [24:0] local_d_2_fu_301_p4;
wire   [24:0] local_d_1_fu_292_p4;
wire   [24:0] local_d_fu_283_p4;
wire   [20:0] local_ug_fu_376_p4;
wire   [20:0] local_ug_1_fu_389_p4;
wire   [20:0] local_ug_2_fu_398_p4;
wire   [20:0] local_ug_3_fu_407_p4;
wire   [20:0] local_ug_4_fu_416_p4;
wire   [20:0] local_ug_5_fu_425_p4;
wire   [20:0] local_ug_6_fu_434_p4;
wire   [20:0] local_ug_7_fu_443_p4;
wire  signed [27:0] sext_ln170_5_fu_472_p1;
wire  signed [27:0] sext_ln170_4_fu_468_p1;
wire  signed [27:0] sext_ln170_3_fu_464_p1;
wire  signed [27:0] sext_ln170_2_fu_460_p1;
wire  signed [27:0] sext_ln170_1_fu_456_p1;
wire  signed [27:0] sext_ln170_fu_452_p1;
wire  signed [27:0] sext_ln168_fu_385_p1;
wire   [216:0] tmp_3_fu_476_p9;
wire   [20:0] local_o_fu_501_p4;
wire  signed [24:0] sext_ln158_5_fu_526_p1;
wire  signed [24:0] sext_ln158_4_fu_523_p1;
wire  signed [24:0] sext_ln158_3_fu_520_p1;
wire  signed [24:0] sext_ln158_2_fu_517_p1;
wire  signed [24:0] sext_ln158_1_fu_514_p1;
wire  signed [24:0] sext_ln161_fu_529_p1;
wire  signed [24:0] sext_ln158_fu_510_p1;
wire   [195:0] tmp_fu_532_p9;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
wire    ap_start_int;
reg    ap_condition_85;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 iter_fu_152 = 13'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
    if ((1'b1 == ap_condition_85)) begin
        if ((icmp_ln150_fu_184_p2 == 1'd0)) begin
            iter_fu_152 <= iter_5_fu_190_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            iter_fu_152 <= 13'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_read_reg_567 <= gemm_stream_TDATA;
        icmp_ln150_reg_563 <= icmp_ln150_fu_184_p2;
        icmp_ln153_reg_623 <= icmp_ln153_fu_266_p2;
        icmp_ln162_reg_627 <= icmp_ln162_fu_272_p2;
        local_o_16_reg_593 <= {{gemm_stream_TDATA[89:69]}};
        local_o_1_reg_598 <= {{gemm_stream_TDATA[119:99]}};
        local_o_2_reg_603 <= {{gemm_stream_TDATA[149:129]}};
        local_o_3_reg_608 <= {{gemm_stream_TDATA[179:159]}};
        local_o_4_reg_613 <= {{gemm_stream_TDATA[209:189]}};
        lshr_ln152_6_reg_618 <= {{gemm_stream_TDATA[239:219]}};
        lshr_ln_reg_588 <= {{gemm_stream_TDATA[59:39]}};
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln150_fu_184_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_iter_4 = 13'd0;
    end else begin
        ap_sig_allocacmp_iter_4 = iter_fu_152;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln150_fu_184_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln150_fu_184_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_predicate_op61_write_state2 == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        mlp1_stream_TDATA_blk_n = mlp1_stream_TREADY;
    end else begin
        mlp1_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_predicate_op61_write_state2 == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        mlp1_stream_TVALID = 1'b1;
    end else begin
        mlp1_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state2_pp0_stage0_iter1))) begin
        if ((ap_predicate_op73_write_state2 == 1'b1)) begin
            od_fc2_stream_TDATA = sext_ln161_1_fu_551_p1;
        end else if ((ap_predicate_op42_write_state2 == 1'b1)) begin
            od_fc2_stream_TDATA = p_s_fu_355_p9;
        end else begin
            od_fc2_stream_TDATA = 'bx;
        end
    end else begin
        od_fc2_stream_TDATA = 'bx;
    end
end

always @ (*) begin
    if ((((ap_predicate_op73_write_state2 == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2)) | ((ap_predicate_op42_write_state2 == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2)))) begin
        od_fc2_stream_TDATA_blk_n = od_fc2_stream_TREADY;
    end else begin
        od_fc2_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_predicate_op73_write_state2 == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (ap_predicate_op42_write_state2 == 1'b1) & (1'b1 == ap_CS_iter1_fsm_state2)))) begin
        od_fc2_stream_TVALID = 1'b1;
    end else begin
        od_fc2_stream_TVALID = 1'b0;
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
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln150_reg_563_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
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

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln150_fu_184_p2 == 1'd0) & (gemm_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = (((ap_predicate_op73_write_state2 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)) | ((ap_predicate_op61_write_state2 == 1'b1) & (mlp1_stream_TREADY == 1'b0)) | ((ap_predicate_op42_write_state2 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = (((ap_predicate_op73_write_state2 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)) | ((ap_predicate_op61_write_state2 == 1'b1) & (mlp1_stream_TREADY == 1'b0)) | ((ap_predicate_op42_write_state2 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_condition_85 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op42_write_state2 = ((icmp_ln162_reg_627 == 1'd0) & (icmp_ln153_reg_623 == 1'd0) & (icmp_ln150_reg_563 == 1'd0));
end

always @ (*) begin
    ap_predicate_op61_write_state2 = ((icmp_ln162_reg_627 == 1'd1) & (icmp_ln153_reg_623 == 1'd0) & (icmp_ln150_reg_563 == 1'd0));
end

always @ (*) begin
    ap_predicate_op73_write_state2 = ((icmp_ln153_reg_623 == 1'd1) & (icmp_ln150_reg_563 == 1'd0));
end

assign icmp_ln150_fu_184_p2 = ((ap_sig_allocacmp_iter_4 == 13'd7040) ? 1'b1 : 1'b0);

assign icmp_ln150_reg_563_pp0_iter0_reg = icmp_ln150_reg_563;

assign icmp_ln153_fu_266_p2 = ((ap_sig_allocacmp_iter_4 < 13'd960) ? 1'b1 : 1'b0);

assign icmp_ln162_fu_272_p2 = ((ap_sig_allocacmp_iter_4 < 13'd6080) ? 1'b1 : 1'b0);

assign iter_5_fu_190_p2 = (ap_sig_allocacmp_iter_4 + 13'd1);

assign local_d_1_fu_292_p4 = {{gemm_stream_read_reg_567[56:32]}};

assign local_d_2_fu_301_p4 = {{gemm_stream_read_reg_567[86:62]}};

assign local_d_3_fu_310_p4 = {{gemm_stream_read_reg_567[116:92]}};

assign local_d_4_fu_319_p4 = {{gemm_stream_read_reg_567[146:122]}};

assign local_d_5_fu_328_p4 = {{gemm_stream_read_reg_567[176:152]}};

assign local_d_6_fu_337_p4 = {{gemm_stream_read_reg_567[206:182]}};

assign local_d_7_fu_346_p4 = {{gemm_stream_read_reg_567[236:212]}};

assign local_d_fu_283_p4 = {{gemm_stream_read_reg_567[26:2]}};

assign local_o_fu_501_p4 = {{gemm_stream_read_reg_567[29:9]}};

assign local_ug_1_fu_389_p4 = {{gemm_stream_read_reg_567[53:33]}};

assign local_ug_2_fu_398_p4 = {{gemm_stream_read_reg_567[83:63]}};

assign local_ug_3_fu_407_p4 = {{gemm_stream_read_reg_567[113:93]}};

assign local_ug_4_fu_416_p4 = {{gemm_stream_read_reg_567[143:123]}};

assign local_ug_5_fu_425_p4 = {{gemm_stream_read_reg_567[173:153]}};

assign local_ug_6_fu_434_p4 = {{gemm_stream_read_reg_567[203:183]}};

assign local_ug_7_fu_443_p4 = {{gemm_stream_read_reg_567[233:213]}};

assign local_ug_fu_376_p4 = {{gemm_stream_read_reg_567[23:3]}};

assign mlp1_stream_TDATA = $signed(tmp_3_fu_476_p9);

assign p_s_fu_355_p9 = {{{{{{{{local_d_7_fu_346_p4}, {local_d_6_fu_337_p4}}, {local_d_5_fu_328_p4}}, {local_d_4_fu_319_p4}}, {local_d_3_fu_310_p4}}, {local_d_2_fu_301_p4}}, {local_d_1_fu_292_p4}}, {local_d_fu_283_p4}};

assign sext_ln158_1_fu_514_p1 = $signed(local_o_16_reg_593);

assign sext_ln158_2_fu_517_p1 = $signed(local_o_1_reg_598);

assign sext_ln158_3_fu_520_p1 = $signed(local_o_2_reg_603);

assign sext_ln158_4_fu_523_p1 = $signed(local_o_3_reg_608);

assign sext_ln158_5_fu_526_p1 = $signed(local_o_4_reg_613);

assign sext_ln158_fu_510_p1 = $signed(local_o_fu_501_p4);

assign sext_ln161_1_fu_551_p1 = $signed(tmp_fu_532_p9);

assign sext_ln161_fu_529_p1 = $signed(lshr_ln_reg_588);

assign sext_ln168_fu_385_p1 = $signed(local_ug_fu_376_p4);

assign sext_ln170_1_fu_456_p1 = $signed(local_ug_2_fu_398_p4);

assign sext_ln170_2_fu_460_p1 = $signed(local_ug_3_fu_407_p4);

assign sext_ln170_3_fu_464_p1 = $signed(local_ug_4_fu_416_p4);

assign sext_ln170_4_fu_468_p1 = $signed(local_ug_5_fu_425_p4);

assign sext_ln170_5_fu_472_p1 = $signed(local_ug_6_fu_434_p4);

assign sext_ln170_fu_452_p1 = $signed(local_ug_1_fu_389_p4);

assign tmp_3_fu_476_p9 = {{{{{{{{local_ug_7_fu_443_p4}, {sext_ln170_5_fu_472_p1}}, {sext_ln170_4_fu_468_p1}}, {sext_ln170_3_fu_464_p1}}, {sext_ln170_2_fu_460_p1}}, {sext_ln170_1_fu_456_p1}}, {sext_ln170_fu_452_p1}}, {sext_ln168_fu_385_p1}};

assign tmp_fu_532_p9 = {{{{{{{{lshr_ln152_6_reg_618}, {sext_ln158_5_fu_526_p1}}, {sext_ln158_4_fu_523_p1}}, {sext_ln158_3_fu_520_p1}}, {sext_ln158_2_fu_517_p1}}, {sext_ln158_1_fu_514_p1}}, {sext_ln161_fu_529_p1}}, {sext_ln158_fu_510_p1}};

endmodule //DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TVALID,
        bias_stream_TVALID,
        qk_stream_TREADY,
        v_stream_TREADY,
        qk_stream_TDATA,
        qk_stream_TVALID,
        gemm_stream_TDATA,
        gemm_stream_TREADY,
        bias_stream_TDATA,
        bias_stream_TREADY,
        v_stream_TDATA,
        v_stream_TVALID
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
input   gemm_stream_TVALID;
input   bias_stream_TVALID;
input   qk_stream_TREADY;
input   v_stream_TREADY;
output  [183:0] qk_stream_TDATA;
output   qk_stream_TVALID;
input  [239:0] gemm_stream_TDATA;
output   gemm_stream_TREADY;
input  [223:0] bias_stream_TDATA;
output   bias_stream_TREADY;
output  [183:0] v_stream_TDATA;
output   v_stream_TVALID;

reg ap_idle;
reg qk_stream_TVALID;
reg gemm_stream_TREADY;
reg bias_stream_TREADY;
reg v_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [1:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
wire   [0:0] icmp_ln224_fu_226_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
reg   [0:0] icmp_ln224_reg_756;
reg   [0:0] icmp_ln224_reg_756_pp0_iter2_reg;
reg   [0:0] tmp_reg_804;
reg   [0:0] tmp_reg_804_pp0_iter2_reg;
reg    ap_predicate_op102_write_state4;
reg    ap_predicate_op104_write_state4;
reg    ap_block_state4_pp0_stage0_iter3;
reg    ap_block_state4_io;
wire    ap_CS_iter3_fsm_state4;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    qk_stream_TDATA_blk_n;
reg    gemm_stream_TDATA_blk_n;
reg    bias_stream_TDATA_blk_n;
reg    v_stream_TDATA_blk_n;
wire   [0:0] icmp_ln224_reg_756_pp0_iter0_reg;
reg   [0:0] icmp_ln224_reg_756_pp0_iter1_reg;
wire   [0:0] icmp_ln225_fu_238_p2;
reg   [0:0] icmp_ln225_reg_760;
reg   [239:0] gemm_stream_read_reg_767;
reg   [239:0] gemm_stream_read_reg_767_pp0_iter1_reg;
reg   [223:0] bias_stream_read_reg_787;
reg   [223:0] bias_stream_read_reg_787_pp0_iter1_reg;
wire   [1:0] select_ln225_fu_304_p3;
reg   [1:0] select_ln225_reg_799;
wire   [19:0] local_qkv_1_fu_389_p2;
reg   [19:0] local_qkv_1_reg_808;
wire   [19:0] local_qkv_3_fu_430_p2;
reg   [19:0] local_qkv_3_reg_813;
wire   [19:0] local_qkv_5_fu_471_p2;
reg   [19:0] local_qkv_5_reg_818;
wire   [19:0] local_qkv_7_fu_512_p2;
reg   [19:0] local_qkv_7_reg_823;
wire   [19:0] local_qkv_9_fu_553_p2;
reg   [19:0] local_qkv_9_reg_828;
wire   [19:0] local_qkv_11_fu_594_p2;
reg   [19:0] local_qkv_11_reg_833;
wire   [19:0] local_qkv_13_fu_635_p2;
reg   [19:0] local_qkv_13_reg_838;
wire   [19:0] local_qkv_15_fu_676_p2;
reg   [19:0] local_qkv_15_reg_843;
reg   [14:0] indvar_flatten12_fu_158;
wire   [14:0] select_ln226_fu_331_p3;
wire    ap_loop_init;
reg   [1:0] qkv_fu_162;
reg   [15:0] indvar_flatten34_fu_166;
wire   [15:0] select_ln225_1_fu_250_p3;
reg   [15:0] ap_sig_allocacmp_indvar_flatten34_load;
reg   [18:0] indvar_flatten64_fu_170;
wire   [18:0] add_ln224_fu_232_p2;
reg   [18:0] ap_sig_allocacmp_indvar_flatten64_load;
wire  signed [183:0] sext_ln248_6_fu_722_p1;
wire   [15:0] add_ln225_1_fu_244_p2;
wire   [0:0] icmp_ln226_fu_286_p2;
wire   [0:0] xor_ln224_fu_281_p2;
wire   [1:0] select_ln224_fu_274_p3;
wire   [0:0] and_ln224_fu_292_p2;
wire   [1:0] qkv_2_fu_298_p2;
wire   [0:0] or_ln226_fu_326_p2;
wire   [14:0] add_ln226_fu_320_p2;
wire   [1:0] empty_fu_349_p2;
wire   [0:0] empty_24_fu_354_p2;
wire   [19:0] tmp_3_fu_360_p4;
wire   [19:0] tmp_4_fu_369_p4;
wire   [19:0] trunc_ln242_fu_386_p1;
wire   [19:0] local_qkv_fu_378_p3;
wire   [19:0] tmp_5_fu_395_p4;
wire   [19:0] tmp_6_fu_404_p4;
wire   [19:0] trunc_ln242_1_fu_421_p4;
wire   [19:0] local_qkv_2_fu_413_p3;
wire   [19:0] tmp_7_fu_436_p4;
wire   [19:0] tmp_8_fu_445_p4;
wire   [19:0] trunc_ln242_2_fu_462_p4;
wire   [19:0] local_qkv_4_fu_454_p3;
wire   [19:0] tmp_9_fu_477_p4;
wire   [19:0] tmp_s_fu_486_p4;
wire   [19:0] trunc_ln242_3_fu_503_p4;
wire   [19:0] local_qkv_6_fu_495_p3;
wire   [19:0] tmp_1_fu_518_p4;
wire   [19:0] tmp_10_fu_527_p4;
wire   [19:0] trunc_ln242_4_fu_544_p4;
wire   [19:0] local_qkv_8_fu_536_p3;
wire   [19:0] tmp_11_fu_559_p4;
wire   [19:0] tmp_12_fu_568_p4;
wire   [19:0] trunc_ln242_5_fu_585_p4;
wire   [19:0] local_qkv_10_fu_577_p3;
wire   [19:0] tmp_13_fu_600_p4;
wire   [19:0] tmp_14_fu_609_p4;
wire   [19:0] trunc_ln242_6_fu_626_p4;
wire   [19:0] local_qkv_12_fu_618_p3;
wire   [19:0] tmp_15_fu_641_p4;
wire   [19:0] tmp_16_fu_650_p4;
wire   [19:0] trunc_ln242_7_fu_667_p4;
wire   [19:0] local_qkv_14_fu_659_p3;
wire  signed [22:0] sext_ln248_5_fu_700_p1;
wire  signed [22:0] sext_ln248_4_fu_697_p1;
wire  signed [22:0] sext_ln248_3_fu_694_p1;
wire  signed [22:0] sext_ln248_2_fu_691_p1;
wire  signed [22:0] sext_ln248_1_fu_688_p1;
wire  signed [22:0] sext_ln248_fu_685_p1;
wire  signed [22:0] sext_ln243_fu_682_p1;
wire   [180:0] tmp_2_fu_703_p9;
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
reg    ap_condition_105;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 indvar_flatten12_fu_158 = 15'd0;
//#0 qkv_fu_162 = 2'd0;
//#0 indvar_flatten34_fu_166 = 16'd0;
//#0 indvar_flatten64_fu_170 = 19'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        indvar_flatten12_fu_158 <= 15'd0;
    end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln224_reg_756_pp0_iter0_reg == 1'd0))) begin
        indvar_flatten12_fu_158 <= select_ln226_fu_331_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_105)) begin
        if ((icmp_ln224_fu_226_p2 == 1'd0)) begin
            indvar_flatten34_fu_166 <= select_ln225_1_fu_250_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten34_fu_166 <= 16'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_105)) begin
        if ((icmp_ln224_fu_226_p2 == 1'd0)) begin
            indvar_flatten64_fu_170 <= add_ln224_fu_232_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten64_fu_170 <= 19'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        qkv_fu_162 <= 2'd0;
    end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln224_reg_756_pp0_iter0_reg == 1'd0))) begin
        qkv_fu_162 <= select_ln225_fu_304_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        bias_stream_read_reg_787 <= bias_stream_TDATA;
        gemm_stream_read_reg_767 <= gemm_stream_TDATA;
        icmp_ln224_reg_756 <= icmp_ln224_fu_226_p2;
        icmp_ln225_reg_760 <= icmp_ln225_fu_238_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        bias_stream_read_reg_787_pp0_iter1_reg <= bias_stream_read_reg_787;
        gemm_stream_read_reg_767_pp0_iter1_reg <= gemm_stream_read_reg_767;
        icmp_ln224_reg_756_pp0_iter1_reg <= icmp_ln224_reg_756;
        select_ln225_reg_799 <= select_ln225_fu_304_p3;
        tmp_reg_804 <= select_ln225_fu_304_p3[32'd1];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        icmp_ln224_reg_756_pp0_iter2_reg <= icmp_ln224_reg_756_pp0_iter1_reg;
        local_qkv_11_reg_833 <= local_qkv_11_fu_594_p2;
        local_qkv_13_reg_838 <= local_qkv_13_fu_635_p2;
        local_qkv_15_reg_843 <= local_qkv_15_fu_676_p2;
        local_qkv_1_reg_808 <= local_qkv_1_fu_389_p2;
        local_qkv_3_reg_813 <= local_qkv_3_fu_430_p2;
        local_qkv_5_reg_818 <= local_qkv_5_fu_471_p2;
        local_qkv_7_reg_823 <= local_qkv_7_fu_512_p2;
        local_qkv_9_reg_828 <= local_qkv_9_fu_553_p2;
        tmp_reg_804_pp0_iter2_reg <= tmp_reg_804;
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
    if (((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) begin
        ap_ST_iter3_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter3_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (icmp_ln224_fu_226_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (ap_loop_exit_ready_pp0_iter3_reg == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten34_load = 16'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten34_load = indvar_flatten34_fu_166;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten64_load = 19'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten64_load = indvar_flatten64_fu_170;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln224_fu_226_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        bias_stream_TDATA_blk_n = bias_stream_TVALID;
    end else begin
        bias_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (icmp_ln224_fu_226_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        bias_stream_TREADY = 1'b1;
    end else begin
        bias_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln224_fu_226_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (icmp_ln224_fu_226_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_predicate_op102_write_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        qk_stream_TDATA_blk_n = qk_stream_TREADY;
    end else begin
        qk_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (ap_predicate_op102_write_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        qk_stream_TVALID = 1'b1;
    end else begin
        qk_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((ap_predicate_op104_write_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        v_stream_TDATA_blk_n = v_stream_TREADY;
    end else begin
        v_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (ap_predicate_op104_write_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        v_stream_TVALID = 1'b1;
    end else begin
        v_stream_TVALID = 1'b0;
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
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if ((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else if (((~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (1'b1 == ap_CS_iter2_fsm_state3)) | (~((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)) & (icmp_ln224_reg_756_pp0_iter2_reg == 1'd1) & (1'b1 == ap_CS_iter3_fsm_state4)))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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

assign add_ln224_fu_232_p2 = (ap_sig_allocacmp_indvar_flatten64_load + 19'd1);

assign add_ln225_1_fu_244_p2 = (ap_sig_allocacmp_indvar_flatten34_load + 16'd1);

assign add_ln226_fu_320_p2 = (indvar_flatten12_fu_158 + 15'd1);

assign and_ln224_fu_292_p2 = (xor_ln224_fu_281_p2 & icmp_ln226_fu_286_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state4 = ap_CS_iter3_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((bias_stream_TVALID == 1'b0) & (icmp_ln224_fu_226_p2 == 1'd0)) | ((icmp_ln224_fu_226_p2 == 1'd0) & (gemm_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state4_io = (((ap_predicate_op104_write_state4 == 1'b1) & (v_stream_TREADY == 1'b0)) | ((ap_predicate_op102_write_state4 == 1'b1) & (qk_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state4_pp0_stage0_iter3 = (((ap_predicate_op104_write_state4 == 1'b1) & (v_stream_TREADY == 1'b0)) | ((ap_predicate_op102_write_state4 == 1'b1) & (qk_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_condition_105 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter3_fsm_state4) & ((1'b1 == ap_block_state4_io) | (1'b1 == ap_block_state4_pp0_stage0_iter3)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op102_write_state4 = ((tmp_reg_804_pp0_iter2_reg == 1'd0) & (icmp_ln224_reg_756_pp0_iter2_reg == 1'd0));
end

always @ (*) begin
    ap_predicate_op104_write_state4 = ((tmp_reg_804_pp0_iter2_reg == 1'd1) & (icmp_ln224_reg_756_pp0_iter2_reg == 1'd0));
end

assign empty_24_fu_354_p2 = ((empty_fu_349_p2 == 2'd1) ? 1'b1 : 1'b0);

assign empty_fu_349_p2 = (select_ln225_reg_799 | 2'd1);

assign icmp_ln224_fu_226_p2 = ((ap_sig_allocacmp_indvar_flatten64_load == 19'd294912) ? 1'b1 : 1'b0);

assign icmp_ln224_reg_756_pp0_iter0_reg = icmp_ln224_reg_756;

assign icmp_ln225_fu_238_p2 = ((ap_sig_allocacmp_indvar_flatten34_load == 16'd24576) ? 1'b1 : 1'b0);

assign icmp_ln226_fu_286_p2 = ((indvar_flatten12_fu_158 == 15'd8192) ? 1'b1 : 1'b0);

assign local_qkv_10_fu_577_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_11_fu_559_p4 : tmp_12_fu_568_p4);

assign local_qkv_11_fu_594_p2 = (trunc_ln242_5_fu_585_p4 + local_qkv_10_fu_577_p3);

assign local_qkv_12_fu_618_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_13_fu_600_p4 : tmp_14_fu_609_p4);

assign local_qkv_13_fu_635_p2 = (trunc_ln242_6_fu_626_p4 + local_qkv_12_fu_618_p3);

assign local_qkv_14_fu_659_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_15_fu_641_p4 : tmp_16_fu_650_p4);

assign local_qkv_15_fu_676_p2 = (trunc_ln242_7_fu_667_p4 + local_qkv_14_fu_659_p3);

assign local_qkv_1_fu_389_p2 = (trunc_ln242_fu_386_p1 + local_qkv_fu_378_p3);

assign local_qkv_2_fu_413_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_5_fu_395_p4 : tmp_6_fu_404_p4);

assign local_qkv_3_fu_430_p2 = (trunc_ln242_1_fu_421_p4 + local_qkv_2_fu_413_p3);

assign local_qkv_4_fu_454_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_7_fu_436_p4 : tmp_8_fu_445_p4);

assign local_qkv_5_fu_471_p2 = (trunc_ln242_2_fu_462_p4 + local_qkv_4_fu_454_p3);

assign local_qkv_6_fu_495_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_9_fu_477_p4 : tmp_s_fu_486_p4);

assign local_qkv_7_fu_512_p2 = (trunc_ln242_3_fu_503_p4 + local_qkv_6_fu_495_p3);

assign local_qkv_8_fu_536_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_1_fu_518_p4 : tmp_10_fu_527_p4);

assign local_qkv_9_fu_553_p2 = (trunc_ln242_4_fu_544_p4 + local_qkv_8_fu_536_p3);

assign local_qkv_fu_378_p3 = ((empty_24_fu_354_p2[0:0] == 1'b1) ? tmp_3_fu_360_p4 : tmp_4_fu_369_p4);

assign or_ln226_fu_326_p2 = (icmp_ln225_reg_760 | and_ln224_fu_292_p2);

assign qk_stream_TDATA = sext_ln248_6_fu_722_p1;

assign qkv_2_fu_298_p2 = (select_ln224_fu_274_p3 + 2'd1);

assign select_ln224_fu_274_p3 = ((icmp_ln225_reg_760[0:0] == 1'b1) ? 2'd0 : qkv_fu_162);

assign select_ln225_1_fu_250_p3 = ((icmp_ln225_fu_238_p2[0:0] == 1'b1) ? 16'd1 : add_ln225_1_fu_244_p2);

assign select_ln225_fu_304_p3 = ((and_ln224_fu_292_p2[0:0] == 1'b1) ? qkv_2_fu_298_p2 : select_ln224_fu_274_p3);

assign select_ln226_fu_331_p3 = ((or_ln226_fu_326_p2[0:0] == 1'b1) ? 15'd1 : add_ln226_fu_320_p2);

assign sext_ln243_fu_682_p1 = $signed(local_qkv_1_reg_808);

assign sext_ln248_1_fu_688_p1 = $signed(local_qkv_5_reg_818);

assign sext_ln248_2_fu_691_p1 = $signed(local_qkv_7_reg_823);

assign sext_ln248_3_fu_694_p1 = $signed(local_qkv_9_reg_828);

assign sext_ln248_4_fu_697_p1 = $signed(local_qkv_11_reg_833);

assign sext_ln248_5_fu_700_p1 = $signed(local_qkv_13_reg_838);

assign sext_ln248_6_fu_722_p1 = $signed(tmp_2_fu_703_p9);

assign sext_ln248_fu_685_p1 = $signed(local_qkv_3_reg_813);

assign tmp_10_fu_527_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[144:125]}};

assign tmp_11_fu_559_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[173:154]}};

assign tmp_12_fu_568_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[174:155]}};

assign tmp_13_fu_600_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[203:184]}};

assign tmp_14_fu_609_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[204:185]}};

assign tmp_15_fu_641_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[233:214]}};

assign tmp_16_fu_650_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[234:215]}};

assign tmp_1_fu_518_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[143:124]}};

assign tmp_2_fu_703_p9 = {{{{{{{{local_qkv_15_reg_843}, {sext_ln248_5_fu_700_p1}}, {sext_ln248_4_fu_697_p1}}, {sext_ln248_3_fu_694_p1}}, {sext_ln248_2_fu_691_p1}}, {sext_ln248_1_fu_688_p1}}, {sext_ln248_fu_685_p1}}, {sext_ln243_fu_682_p1}};

assign tmp_3_fu_360_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[23:4]}};

assign tmp_4_fu_369_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[24:5]}};

assign tmp_5_fu_395_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[53:34]}};

assign tmp_6_fu_404_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[54:35]}};

assign tmp_7_fu_436_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[83:64]}};

assign tmp_8_fu_445_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[84:65]}};

assign tmp_9_fu_477_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[113:94]}};

assign tmp_s_fu_486_p4 = {{gemm_stream_read_reg_767_pp0_iter1_reg[114:95]}};

assign trunc_ln242_1_fu_421_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[47:28]}};

assign trunc_ln242_2_fu_462_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[75:56]}};

assign trunc_ln242_3_fu_503_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[103:84]}};

assign trunc_ln242_4_fu_544_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[131:112]}};

assign trunc_ln242_5_fu_585_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[159:140]}};

assign trunc_ln242_6_fu_626_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[187:168]}};

assign trunc_ln242_7_fu_667_p4 = {{bias_stream_read_reg_787_pp0_iter1_reg[215:196]}};

assign trunc_ln242_fu_386_p1 = bias_stream_read_reg_787_pp0_iter1_reg[19:0];

assign v_stream_TDATA = sext_ln248_6_fu_722_p1;

assign xor_ln224_fu_281_p2 = (icmp_ln225_reg_760 ^ 1'd1);

endmodule //DEMUX_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_vit_encoder (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TDATA,
        gemm_stream_TVALID,
        gemm_stream_TREADY,
        bias_stream_TDATA,
        bias_stream_TVALID,
        bias_stream_TREADY,
        qk_stream_TDATA,
        qk_stream_TVALID,
        qk_stream_TREADY,
        v_stream_TDATA,
        v_stream_TVALID,
        v_stream_TREADY,
        mlp1_stream_TDATA,
        mlp1_stream_TVALID,
        mlp1_stream_TREADY,
        od_fc2_stream_TDATA,
        od_fc2_stream_TVALID,
        od_fc2_stream_TREADY
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
input  [239:0] gemm_stream_TDATA;
input   gemm_stream_TVALID;
output   gemm_stream_TREADY;
input  [223:0] bias_stream_TDATA;
input   bias_stream_TVALID;
output   bias_stream_TREADY;
output  [183:0] qk_stream_TDATA;
output   qk_stream_TVALID;
input   qk_stream_TREADY;
output  [183:0] v_stream_TDATA;
output   v_stream_TVALID;
input   v_stream_TREADY;
output  [223:0] mlp1_stream_TDATA;
output   mlp1_stream_TVALID;
input   mlp1_stream_TREADY;
output  [199:0] od_fc2_stream_TDATA;
output   od_fc2_stream_TVALID;
input   od_fc2_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg gemm_stream_TREADY;
reg bias_stream_TREADY;

(* fsm_encoding = "none" *) reg   [4:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_done;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_idle;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_ready;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TREADY;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TREADY;
wire   [183:0] grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TDATA;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TVALID;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_gemm_stream_TREADY;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_bias_stream_TREADY;
wire   [183:0] grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TDATA;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TVALID;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_done;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_idle;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_ready;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TREADY;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TREADY;
wire   [199:0] grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TDATA;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TVALID;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_gemm_stream_TREADY;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_bias_stream_TREADY;
wire   [223:0] grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TDATA;
wire    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TVALID;
reg    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start_reg;
wire    ap_CS_fsm_state2;
reg    grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start_reg;
wire    ap_CS_fsm_state4;
wire    ap_CS_fsm_state5;
reg   [4:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 5'd1;
//#0 grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start_reg = 1'b0;
//#0 grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start_reg = 1'b0;
end

DEMUX_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start),
    .ap_done(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_done),
    .ap_idle(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_idle),
    .ap_ready(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_ready),
    .gemm_stream_TVALID(gemm_stream_TVALID),
    .bias_stream_TVALID(bias_stream_TVALID),
    .qk_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TREADY),
    .v_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TREADY),
    .qk_stream_TDATA(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TDATA),
    .qk_stream_TVALID(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TVALID),
    .gemm_stream_TDATA(gemm_stream_TDATA),
    .gemm_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_gemm_stream_TREADY),
    .bias_stream_TDATA(bias_stream_TDATA),
    .bias_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_bias_stream_TREADY),
    .v_stream_TDATA(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TDATA),
    .v_stream_TVALID(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TVALID)
);

DEMUX_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7 grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start),
    .ap_done(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_done),
    .ap_idle(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_idle),
    .ap_ready(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_ready),
    .gemm_stream_TVALID(gemm_stream_TVALID),
    .bias_stream_TVALID(bias_stream_TVALID),
    .od_fc2_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TREADY),
    .mlp1_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TREADY),
    .od_fc2_stream_TDATA(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TDATA),
    .od_fc2_stream_TVALID(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TVALID),
    .gemm_stream_TDATA(gemm_stream_TDATA),
    .gemm_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_gemm_stream_TREADY),
    .bias_stream_TDATA(bias_stream_TDATA),
    .bias_stream_TREADY(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_bias_stream_TREADY),
    .mlp1_stream_TDATA(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TDATA),
    .mlp1_stream_TVALID(grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TVALID)
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
        grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
            grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start_reg <= 1'b1;
        end else if ((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_ready == 1'b1)) begin
            grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state4)) begin
            grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start_reg <= 1'b1;
        end else if ((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_ready == 1'b1)) begin
            grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start_reg <= 1'b0;
        end
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_done == 1'b0)) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_done == 1'b0)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) | ((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state5)))) begin
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
    if (((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state5))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        bias_stream_TREADY = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_bias_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state2)) begin
        bias_stream_TREADY = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_bias_stream_TREADY;
    end else begin
        bias_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        gemm_stream_TREADY = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_gemm_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state2)) begin
        gemm_stream_TREADY = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_gemm_stream_TREADY;
    end else begin
        gemm_stream_TREADY = 1'b0;
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
            if (((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state5))) begin
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

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_ap_start_reg;

assign grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TREADY = (qk_stream_TREADY & ap_CS_fsm_state2);

assign grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TREADY = (v_stream_TREADY & ap_CS_fsm_state2);

assign grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_ap_start_reg;

assign grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TREADY = (mlp1_stream_TREADY & ap_CS_fsm_state5);

assign grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TREADY = (od_fc2_stream_TREADY & ap_CS_fsm_state5);

assign mlp1_stream_TDATA = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TDATA;

assign mlp1_stream_TVALID = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_mlp1_stream_TVALID;

assign od_fc2_stream_TDATA = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TDATA;

assign od_fc2_stream_TVALID = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7_fu_42_od_fc2_stream_TVALID;

assign qk_stream_TDATA = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TDATA;

assign qk_stream_TVALID = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_qk_stream_TVALID;

assign v_stream_TDATA = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TDATA;

assign v_stream_TVALID = grp_demux_vit_encoder_Pipeline_VITIS_LOOP_224_1_VITIS_LOOP_225_2_VITIS_LOOP_226_3_VI_fu_30_v_stream_TVALID;

endmodule //DEMUX_demux_vit_encoder
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_llm_decoder (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        chunk_pos,
        gemm_stream_TDATA,
        gemm_stream_TVALID,
        gemm_stream_TREADY,
        qk_stream_TDATA,
        qk_stream_TVALID,
        qk_stream_TREADY,
        v_stream_TDATA,
        v_stream_TVALID,
        v_stream_TREADY,
        mlp1_stream_TDATA,
        mlp1_stream_TVALID,
        mlp1_stream_TREADY,
        od_fc2_stream_TDATA,
        od_fc2_stream_TVALID,
        od_fc2_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 10'd1;
parameter    ap_ST_fsm_state2 = 10'd2;
parameter    ap_ST_fsm_state3 = 10'd4;
parameter    ap_ST_fsm_state4 = 10'd8;
parameter    ap_ST_fsm_state5 = 10'd16;
parameter    ap_ST_fsm_state6 = 10'd32;
parameter    ap_ST_fsm_state7 = 10'd64;
parameter    ap_ST_fsm_state8 = 10'd128;
parameter    ap_ST_fsm_state9 = 10'd256;
parameter    ap_ST_fsm_state10 = 10'd512;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [2:0] chunk_pos;
input  [239:0] gemm_stream_TDATA;
input   gemm_stream_TVALID;
output   gemm_stream_TREADY;
output  [183:0] qk_stream_TDATA;
output   qk_stream_TVALID;
input   qk_stream_TREADY;
output  [183:0] v_stream_TDATA;
output   v_stream_TVALID;
input   v_stream_TREADY;
output  [223:0] mlp1_stream_TDATA;
output   mlp1_stream_TVALID;
input   mlp1_stream_TREADY;
output  [199:0] od_fc2_stream_TDATA;
output   od_fc2_stream_TVALID;
input   od_fc2_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg gemm_stream_TREADY;
reg[183:0] qk_stream_TDATA;
reg qk_stream_TVALID;

(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_done;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_idle;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_ready;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TREADY;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_gemm_stream_TREADY;
wire   [183:0] grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TDATA;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TVALID;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_done;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_idle;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_ready;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TREADY;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TREADY;
wire   [199:0] grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TDATA;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TVALID;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_gemm_stream_TREADY;
wire   [223:0] grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TDATA;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TVALID;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_done;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_idle;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_ready;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TREADY;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_gemm_stream_TREADY;
wire   [183:0] grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TDATA;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TVALID;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_done;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_idle;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_ready;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TREADY;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_gemm_stream_TREADY;
wire   [183:0] grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TDATA;
wire    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TVALID;
reg    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start_reg;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln94_fu_105_p2;
wire    ap_CS_fsm_state3;
reg    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start_reg;
wire    ap_CS_fsm_state10;
reg    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start_reg;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state6;
reg    grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start_reg;
wire    ap_CS_fsm_state8;
wire    ap_CS_fsm_state9;
reg   [2:0] kv_fu_52;
wire   [2:0] kv_2_fu_111_p2;
reg   [9:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
reg    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
reg    ap_ST_fsm_state9_blk;
reg    ap_ST_fsm_state10_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 10'd1;
//#0 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start_reg = 1'b0;
//#0 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start_reg = 1'b0;
//#0 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start_reg = 1'b0;
//#0 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start_reg = 1'b0;
//#0 kv_fu_52 = 3'd0;
end

DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start),
    .ap_done(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_done),
    .ap_idle(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_idle),
    .ap_ready(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_ready),
    .gemm_stream_TVALID(gemm_stream_TVALID),
    .qk_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TREADY),
    .gemm_stream_TDATA(gemm_stream_TDATA),
    .gemm_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_gemm_stream_TREADY),
    .qk_stream_TDATA(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TDATA),
    .qk_stream_TVALID(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TVALID)
);

DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start),
    .ap_done(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_done),
    .ap_idle(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_idle),
    .ap_ready(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_ready),
    .gemm_stream_TVALID(gemm_stream_TVALID),
    .od_fc2_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TREADY),
    .mlp1_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TREADY),
    .od_fc2_stream_TDATA(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TDATA),
    .od_fc2_stream_TVALID(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TVALID),
    .gemm_stream_TDATA(gemm_stream_TDATA),
    .gemm_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_gemm_stream_TREADY),
    .mlp1_stream_TDATA(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TDATA),
    .mlp1_stream_TVALID(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TVALID)
);

DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start),
    .ap_done(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_done),
    .ap_idle(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_idle),
    .ap_ready(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_ready),
    .gemm_stream_TVALID(gemm_stream_TVALID),
    .v_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TREADY),
    .gemm_stream_TDATA(gemm_stream_TDATA),
    .gemm_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_gemm_stream_TREADY),
    .chunk_pos_cast(chunk_pos),
    .v_stream_TDATA(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TDATA),
    .v_stream_TVALID(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TVALID)
);

DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10 grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start),
    .ap_done(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_done),
    .ap_idle(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_idle),
    .ap_ready(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_ready),
    .gemm_stream_TVALID(gemm_stream_TVALID),
    .qk_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TREADY),
    .gemm_stream_TDATA(gemm_stream_TDATA),
    .gemm_stream_TREADY(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_gemm_stream_TREADY),
    .qk_stream_TDATA(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TDATA),
    .qk_stream_TVALID(grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TVALID)
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
        grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state5)) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_ready == 1'b1)) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state8)) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_ready == 1'b1)) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln94_fu_105_p2 == 1'd1))) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_ready == 1'b1)) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln94_fu_105_p2 == 1'd0))) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_ready == 1'b1)) begin
            grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        kv_fu_52 <= 3'd0;
    end else if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln94_fu_105_p2 == 1'd0))) begin
        kv_fu_52 <= kv_2_fu_111_p2;
    end
end

always @ (*) begin
    if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_done == 1'b0)) begin
        ap_ST_fsm_state10_blk = 1'b1;
    end else begin
        ap_ST_fsm_state10_blk = 1'b0;
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
    if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

always @ (*) begin
    if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_done == 1'b0)) begin
        ap_ST_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_fsm_state6_blk = 1'b0;
    end
end

assign ap_ST_fsm_state7_blk = 1'b0;

assign ap_ST_fsm_state8_blk = 1'b0;

always @ (*) begin
    if ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_done == 1'b0)) begin
        ap_ST_fsm_state9_blk = 1'b1;
    end else begin
        ap_ST_fsm_state9_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) | ((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10)))) begin
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
    if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        gemm_stream_TREADY = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_gemm_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        gemm_stream_TREADY = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_gemm_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        gemm_stream_TREADY = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_gemm_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        gemm_stream_TREADY = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_gemm_stream_TREADY;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state9))) begin
        qk_stream_TDATA = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TDATA;
    end else if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
        qk_stream_TDATA = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TDATA;
    end else begin
        qk_stream_TDATA = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        qk_stream_TVALID = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        qk_stream_TVALID = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TVALID;
    end else begin
        qk_stream_TVALID = 1'b0;
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
            if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln94_fu_105_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
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
            if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state6))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state9))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end
        end
        ap_ST_fsm_state10 : begin
            if (((grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_ap_start_reg;

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TREADY = (v_stream_TREADY & ap_CS_fsm_state6);

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_ap_start_reg;

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10_fu_89_qk_stream_TREADY = (qk_stream_TREADY & ap_CS_fsm_state9);

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_ap_start_reg;

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TREADY = (mlp1_stream_TREADY & ap_CS_fsm_state10);

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TREADY = (od_fc2_stream_TREADY & ap_CS_fsm_state10);

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_ap_start_reg;

assign grp_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3_fu_62_qk_stream_TREADY = (qk_stream_TREADY & ap_CS_fsm_state3);

assign icmp_ln94_fu_105_p2 = ((kv_fu_52 == 3'd5) ? 1'b1 : 1'b0);

assign kv_2_fu_111_p2 = (kv_fu_52 + 3'd1);

assign mlp1_stream_TDATA = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TDATA;

assign mlp1_stream_TVALID = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_mlp1_stream_TVALID;

assign od_fc2_stream_TDATA = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TDATA;

assign od_fc2_stream_TVALID = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_150_12_fu_70_od_fc2_stream_TVALID;

assign v_stream_TDATA = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TDATA;

assign v_stream_TVALID = grp_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6_fu_80_v_stream_TVALID;

endmodule //DEMUX_demux_llm_decoder
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TVALID,
        bias_stream_TVALID,
        od_fc2_stream_TREADY,
        mlp1_stream_TREADY,
        od_fc2_stream_TDATA,
        od_fc2_stream_TVALID,
        gemm_stream_TDATA,
        gemm_stream_TREADY,
        bias_stream_TDATA,
        bias_stream_TREADY,
        mlp1_stream_TDATA,
        mlp1_stream_TVALID
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
input   gemm_stream_TVALID;
input   bias_stream_TVALID;
input   od_fc2_stream_TREADY;
input   mlp1_stream_TREADY;
output  [199:0] od_fc2_stream_TDATA;
output   od_fc2_stream_TVALID;
input  [239:0] gemm_stream_TDATA;
output   gemm_stream_TREADY;
input  [223:0] bias_stream_TDATA;
output   bias_stream_TREADY;
output  [223:0] mlp1_stream_TDATA;
output   mlp1_stream_TVALID;

reg ap_idle;
reg[199:0] od_fc2_stream_TDATA;
reg od_fc2_stream_TVALID;
reg gemm_stream_TREADY;
reg bias_stream_TREADY;
reg mlp1_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire   [0:0] icmp_ln259_fu_263_p2;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
reg   [0:0] icmp_ln259_reg_898;
reg   [0:0] icmp_ln259_reg_898_pp0_iter1_reg;
reg   [0:0] icmp_ln263_reg_983;
reg   [0:0] icmp_ln263_reg_983_pp0_iter1_reg;
reg   [0:0] icmp_ln275_reg_987;
reg   [0:0] icmp_ln275_reg_987_pp0_iter1_reg;
reg    ap_predicate_op126_write_state3;
reg    ap_predicate_op129_write_state3;
reg    ap_predicate_op140_write_state3;
reg    ap_block_state3_pp0_stage0_iter2;
reg    ap_block_state3_io;
wire    ap_CS_iter2_fsm_state3;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    od_fc2_stream_TDATA_blk_n;
reg    gemm_stream_TDATA_blk_n;
reg    bias_stream_TDATA_blk_n;
reg    mlp1_stream_TDATA_blk_n;
reg   [239:0] gemm_stream_read_reg_902;
reg   [223:0] bias_stream_read_reg_930;
wire   [27:0] trunc_ln262_fu_269_p1;
reg   [27:0] trunc_ln262_reg_943;
reg   [27:0] trunc_ln262_1_reg_948;
reg   [27:0] trunc_ln262_2_reg_953;
reg   [27:0] trunc_ln262_3_reg_958;
reg   [27:0] trunc_ln262_4_reg_963;
reg   [27:0] trunc_ln262_5_reg_968;
reg   [27:0] trunc_ln262_6_reg_973;
reg   [27:0] trunc_ln262_7_reg_978;
wire   [0:0] icmp_ln263_fu_343_p2;
wire   [0:0] icmp_ln275_fu_349_p2;
wire   [22:0] local_fc2_1_fu_382_p2;
reg   [22:0] local_fc2_1_reg_991;
wire   [22:0] local_fc2_3_fu_401_p2;
reg   [22:0] local_fc2_3_reg_996;
wire   [22:0] local_fc2_5_fu_420_p2;
reg   [22:0] local_fc2_5_reg_1001;
wire   [22:0] local_fc2_7_fu_439_p2;
reg   [22:0] local_fc2_7_reg_1006;
wire   [22:0] local_fc2_9_fu_458_p2;
reg   [22:0] local_fc2_9_reg_1011;
wire   [22:0] local_fc2_11_fu_477_p2;
reg   [22:0] local_fc2_11_reg_1016;
wire   [22:0] local_fc2_13_fu_496_p2;
reg   [22:0] local_fc2_13_reg_1021;
wire   [22:0] local_fc2_15_fu_515_p2;
reg   [22:0] local_fc2_15_reg_1026;
wire   [27:0] local_fc1_1_fu_530_p2;
reg   [27:0] local_fc1_1_reg_1031;
wire   [27:0] local_fc1_3_fu_544_p2;
reg   [27:0] local_fc1_3_reg_1036;
wire   [27:0] local_fc1_5_fu_558_p2;
reg   [27:0] local_fc1_5_reg_1041;
wire   [27:0] local_fc1_7_fu_572_p2;
reg   [27:0] local_fc1_7_reg_1046;
wire   [27:0] local_fc1_9_fu_586_p2;
reg   [27:0] local_fc1_9_reg_1051;
wire   [27:0] local_fc1_11_fu_600_p2;
reg   [27:0] local_fc1_11_reg_1056;
wire   [27:0] local_fc1_13_fu_614_p2;
reg   [27:0] local_fc1_13_reg_1061;
wire   [27:0] local_fc1_15_fu_628_p2;
reg   [27:0] local_fc1_15_reg_1066;
wire   [22:0] local_o_1_fu_649_p2;
reg   [22:0] local_o_1_reg_1071;
wire   [22:0] local_o_3_fu_668_p2;
reg   [22:0] local_o_3_reg_1076;
wire   [22:0] local_o_5_fu_687_p2;
reg   [22:0] local_o_5_reg_1081;
wire   [22:0] local_o_7_fu_706_p2;
reg   [22:0] local_o_7_reg_1086;
wire   [22:0] local_o_9_fu_725_p2;
reg   [22:0] local_o_9_reg_1091;
wire   [22:0] local_o_11_fu_744_p2;
reg   [22:0] local_o_11_reg_1096;
wire   [22:0] local_o_13_fu_763_p2;
reg   [22:0] local_o_13_reg_1101;
wire   [22:0] local_o_15_fu_782_p2;
reg   [22:0] local_o_15_reg_1106;
reg   [19:0] iter_fu_162;
wire   [19:0] iter_3_fu_355_p2;
wire    ap_loop_init;
reg   [19:0] ap_sig_allocacmp_iter_2;
wire  signed [199:0] sext_ln298_6_fu_828_p1;
wire  signed [199:0] sext_ln274_6_fu_886_p1;
wire   [20:0] local_fc2_fu_366_p4;
wire   [22:0] trunc_ln295_fu_379_p1;
wire  signed [22:0] sext_ln294_fu_375_p1;
wire   [20:0] local_fc2_2_fu_388_p4;
wire   [22:0] grp_fu_192_p4;
wire  signed [22:0] sext_ln294_1_fu_397_p1;
wire   [20:0] local_fc2_4_fu_407_p4;
wire   [22:0] grp_fu_201_p4;
wire  signed [22:0] sext_ln294_2_fu_416_p1;
wire   [20:0] local_fc2_6_fu_426_p4;
wire   [22:0] grp_fu_210_p4;
wire  signed [22:0] sext_ln294_3_fu_435_p1;
wire   [20:0] local_fc2_8_fu_445_p4;
wire   [22:0] grp_fu_219_p4;
wire  signed [22:0] sext_ln294_4_fu_454_p1;
wire   [20:0] local_fc2_10_fu_464_p4;
wire   [22:0] grp_fu_228_p4;
wire  signed [22:0] sext_ln294_5_fu_473_p1;
wire   [20:0] local_fc2_12_fu_483_p4;
wire   [22:0] grp_fu_237_p4;
wire  signed [22:0] sext_ln294_6_fu_492_p1;
wire   [20:0] local_fc2_14_fu_502_p4;
wire   [22:0] grp_fu_246_p4;
wire  signed [22:0] sext_ln294_7_fu_511_p1;
wire   [27:0] local_fc1_fu_521_p4;
wire   [27:0] local_fc1_2_fu_535_p4;
wire   [27:0] local_fc1_4_fu_549_p4;
wire   [27:0] local_fc1_6_fu_563_p4;
wire   [27:0] local_fc1_8_fu_577_p4;
wire   [27:0] local_fc1_10_fu_591_p4;
wire   [27:0] local_fc1_12_fu_605_p4;
wire   [27:0] local_fc1_14_fu_619_p4;
wire   [18:0] local_o_fu_633_p4;
wire   [22:0] trunc_ln271_fu_646_p1;
wire  signed [22:0] sext_ln270_fu_642_p1;
wire   [18:0] local_o_2_fu_655_p4;
wire  signed [22:0] sext_ln270_1_fu_664_p1;
wire   [18:0] local_o_4_fu_674_p4;
wire  signed [22:0] sext_ln270_2_fu_683_p1;
wire   [18:0] local_o_6_fu_693_p4;
wire  signed [22:0] sext_ln270_3_fu_702_p1;
wire   [18:0] local_o_8_fu_712_p4;
wire  signed [22:0] sext_ln270_4_fu_721_p1;
wire   [18:0] local_o_10_fu_731_p4;
wire  signed [22:0] sext_ln270_5_fu_740_p1;
wire   [18:0] local_o_12_fu_750_p4;
wire  signed [22:0] sext_ln270_6_fu_759_p1;
wire   [18:0] local_o_14_fu_769_p4;
wire  signed [22:0] sext_ln270_7_fu_778_p1;
wire  signed [24:0] sext_ln298_5_fu_806_p1;
wire  signed [24:0] sext_ln298_4_fu_803_p1;
wire  signed [24:0] sext_ln298_3_fu_800_p1;
wire  signed [24:0] sext_ln298_2_fu_797_p1;
wire  signed [24:0] sext_ln298_1_fu_794_p1;
wire  signed [24:0] sext_ln298_fu_791_p1;
wire  signed [24:0] sext_ln296_fu_788_p1;
wire   [197:0] tmp_1_fu_809_p9;
wire  signed [24:0] sext_ln274_5_fu_864_p1;
wire  signed [24:0] sext_ln274_4_fu_861_p1;
wire  signed [24:0] sext_ln274_3_fu_858_p1;
wire  signed [24:0] sext_ln274_2_fu_855_p1;
wire  signed [24:0] sext_ln274_1_fu_852_p1;
wire  signed [24:0] sext_ln274_fu_849_p1;
wire  signed [24:0] sext_ln272_fu_846_p1;
wire   [197:0] tmp_fu_867_p9;
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
reg    ap_condition_102;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 iter_fu_162 = 20'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_102)) begin
        if ((icmp_ln259_fu_263_p2 == 1'd0)) begin
            iter_fu_162 <= iter_3_fu_355_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            iter_fu_162 <= 20'd294912;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        bias_stream_read_reg_930 <= bias_stream_TDATA;
        gemm_stream_read_reg_902 <= gemm_stream_TDATA;
        icmp_ln259_reg_898 <= icmp_ln259_fu_263_p2;
        icmp_ln263_reg_983 <= icmp_ln263_fu_343_p2;
        icmp_ln275_reg_987 <= icmp_ln275_fu_349_p2;
        trunc_ln262_1_reg_948 <= {{bias_stream_TDATA[55:28]}};
        trunc_ln262_2_reg_953 <= {{bias_stream_TDATA[83:56]}};
        trunc_ln262_3_reg_958 <= {{bias_stream_TDATA[111:84]}};
        trunc_ln262_4_reg_963 <= {{bias_stream_TDATA[139:112]}};
        trunc_ln262_5_reg_968 <= {{bias_stream_TDATA[167:140]}};
        trunc_ln262_6_reg_973 <= {{bias_stream_TDATA[195:168]}};
        trunc_ln262_7_reg_978 <= {{bias_stream_TDATA[223:196]}};
        trunc_ln262_reg_943 <= trunc_ln262_fu_269_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        icmp_ln259_reg_898_pp0_iter1_reg <= icmp_ln259_reg_898;
        icmp_ln263_reg_983_pp0_iter1_reg <= icmp_ln263_reg_983;
        icmp_ln275_reg_987_pp0_iter1_reg <= icmp_ln275_reg_987;
        local_fc1_11_reg_1056 <= local_fc1_11_fu_600_p2;
        local_fc1_13_reg_1061 <= local_fc1_13_fu_614_p2;
        local_fc1_15_reg_1066 <= local_fc1_15_fu_628_p2;
        local_fc1_1_reg_1031 <= local_fc1_1_fu_530_p2;
        local_fc1_3_reg_1036 <= local_fc1_3_fu_544_p2;
        local_fc1_5_reg_1041 <= local_fc1_5_fu_558_p2;
        local_fc1_7_reg_1046 <= local_fc1_7_fu_572_p2;
        local_fc1_9_reg_1051 <= local_fc1_9_fu_586_p2;
        local_fc2_11_reg_1016 <= local_fc2_11_fu_477_p2;
        local_fc2_13_reg_1021 <= local_fc2_13_fu_496_p2;
        local_fc2_15_reg_1026 <= local_fc2_15_fu_515_p2;
        local_fc2_1_reg_991 <= local_fc2_1_fu_382_p2;
        local_fc2_3_reg_996 <= local_fc2_3_fu_401_p2;
        local_fc2_5_reg_1001 <= local_fc2_5_fu_420_p2;
        local_fc2_7_reg_1006 <= local_fc2_7_fu_439_p2;
        local_fc2_9_reg_1011 <= local_fc2_9_fu_458_p2;
        local_o_11_reg_1096 <= local_o_11_fu_744_p2;
        local_o_13_reg_1101 <= local_o_13_fu_763_p2;
        local_o_15_reg_1106 <= local_o_15_fu_782_p2;
        local_o_1_reg_1071 <= local_o_1_fu_649_p2;
        local_o_3_reg_1076 <= local_o_3_fu_668_p2;
        local_o_5_reg_1081 <= local_o_5_fu_687_p2;
        local_o_7_reg_1086 <= local_o_7_fu_706_p2;
        local_o_9_reg_1091 <= local_o_9_fu_725_p2;
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln259_fu_263_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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
        ap_sig_allocacmp_iter_2 = 20'd294912;
    end else begin
        ap_sig_allocacmp_iter_2 = iter_fu_162;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln259_fu_263_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        bias_stream_TDATA_blk_n = bias_stream_TVALID;
    end else begin
        bias_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln259_fu_263_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        bias_stream_TREADY = 1'b1;
    end else begin
        bias_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b1) & (icmp_ln259_fu_263_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (icmp_ln259_fu_263_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_predicate_op129_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        mlp1_stream_TDATA_blk_n = mlp1_stream_TREADY;
    end else begin
        mlp1_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_predicate_op129_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        mlp1_stream_TVALID = 1'b1;
    end else begin
        mlp1_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_pp0_stage0_iter2))) begin
        if ((ap_predicate_op140_write_state3 == 1'b1)) begin
            od_fc2_stream_TDATA = sext_ln274_6_fu_886_p1;
        end else if ((ap_predicate_op126_write_state3 == 1'b1)) begin
            od_fc2_stream_TDATA = sext_ln298_6_fu_828_p1;
        end else begin
            od_fc2_stream_TDATA = 'bx;
        end
    end else begin
        od_fc2_stream_TDATA = 'bx;
    end
end

always @ (*) begin
    if ((((ap_predicate_op140_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3)) | ((ap_predicate_op126_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        od_fc2_stream_TDATA_blk_n = od_fc2_stream_TREADY;
    end else begin
        od_fc2_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_predicate_op140_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3)) | (~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (ap_predicate_op126_write_state3 == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state3)))) begin
        od_fc2_stream_TVALID = 1'b1;
    end else begin
        od_fc2_stream_TVALID = 1'b0;
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
            end else if (((~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (icmp_ln259_reg_898_pp0_iter1_reg == 1'd1) & (1'b1 == ap_CS_iter2_fsm_state3)) | (~((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)) & (1'b1 == ap_CS_iter1_fsm_state2)))) begin
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

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((bias_stream_TVALID == 1'b0) & (icmp_ln259_fu_263_p2 == 1'd0)) | ((icmp_ln259_fu_263_p2 == 1'd0) & (gemm_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state3_io = (((ap_predicate_op140_write_state3 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)) | ((ap_predicate_op129_write_state3 == 1'b1) & (mlp1_stream_TREADY == 1'b0)) | ((ap_predicate_op126_write_state3 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state3_pp0_stage0_iter2 = (((ap_predicate_op140_write_state3 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)) | ((ap_predicate_op129_write_state3 == 1'b1) & (mlp1_stream_TREADY == 1'b0)) | ((ap_predicate_op126_write_state3 == 1'b1) & (od_fc2_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_condition_102 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state3) & ((1'b1 == ap_block_state3_io) | (1'b1 == ap_block_state3_pp0_stage0_iter2)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op126_write_state3 = ((icmp_ln275_reg_987_pp0_iter1_reg == 1'd0) & (icmp_ln263_reg_983_pp0_iter1_reg == 1'd0) & (icmp_ln259_reg_898_pp0_iter1_reg == 1'd0));
end

always @ (*) begin
    ap_predicate_op129_write_state3 = ((icmp_ln275_reg_987_pp0_iter1_reg == 1'd1) & (icmp_ln263_reg_983_pp0_iter1_reg == 1'd0) & (icmp_ln259_reg_898_pp0_iter1_reg == 1'd0));
end

always @ (*) begin
    ap_predicate_op140_write_state3 = ((icmp_ln263_reg_983_pp0_iter1_reg == 1'd1) & (icmp_ln259_reg_898_pp0_iter1_reg == 1'd0));
end

assign grp_fu_192_p4 = {{bias_stream_read_reg_930[50:28]}};

assign grp_fu_201_p4 = {{bias_stream_read_reg_930[78:56]}};

assign grp_fu_210_p4 = {{bias_stream_read_reg_930[106:84]}};

assign grp_fu_219_p4 = {{bias_stream_read_reg_930[134:112]}};

assign grp_fu_228_p4 = {{bias_stream_read_reg_930[162:140]}};

assign grp_fu_237_p4 = {{bias_stream_read_reg_930[190:168]}};

assign grp_fu_246_p4 = {{bias_stream_read_reg_930[218:196]}};

assign icmp_ln259_fu_263_p2 = ((ap_sig_allocacmp_iter_2 == 20'd884736) ? 1'b1 : 1'b0);

assign icmp_ln263_fu_343_p2 = ((ap_sig_allocacmp_iter_2 < 20'd393216) ? 1'b1 : 1'b0);

assign icmp_ln275_fu_349_p2 = ((ap_sig_allocacmp_iter_2 < 20'd786432) ? 1'b1 : 1'b0);

assign iter_3_fu_355_p2 = (ap_sig_allocacmp_iter_2 + 20'd1);

assign local_fc1_10_fu_591_p4 = {{gemm_stream_read_reg_902[178:151]}};

assign local_fc1_11_fu_600_p2 = (trunc_ln262_5_reg_968 + local_fc1_10_fu_591_p4);

assign local_fc1_12_fu_605_p4 = {{gemm_stream_read_reg_902[208:181]}};

assign local_fc1_13_fu_614_p2 = (trunc_ln262_6_reg_973 + local_fc1_12_fu_605_p4);

assign local_fc1_14_fu_619_p4 = {{gemm_stream_read_reg_902[238:211]}};

assign local_fc1_15_fu_628_p2 = (trunc_ln262_7_reg_978 + local_fc1_14_fu_619_p4);

assign local_fc1_1_fu_530_p2 = (trunc_ln262_reg_943 + local_fc1_fu_521_p4);

assign local_fc1_2_fu_535_p4 = {{gemm_stream_read_reg_902[58:31]}};

assign local_fc1_3_fu_544_p2 = (trunc_ln262_1_reg_948 + local_fc1_2_fu_535_p4);

assign local_fc1_4_fu_549_p4 = {{gemm_stream_read_reg_902[88:61]}};

assign local_fc1_5_fu_558_p2 = (trunc_ln262_2_reg_953 + local_fc1_4_fu_549_p4);

assign local_fc1_6_fu_563_p4 = {{gemm_stream_read_reg_902[118:91]}};

assign local_fc1_7_fu_572_p2 = (trunc_ln262_3_reg_958 + local_fc1_6_fu_563_p4);

assign local_fc1_8_fu_577_p4 = {{gemm_stream_read_reg_902[148:121]}};

assign local_fc1_9_fu_586_p2 = (trunc_ln262_4_reg_963 + local_fc1_8_fu_577_p4);

assign local_fc1_fu_521_p4 = {{gemm_stream_read_reg_902[28:1]}};

assign local_fc2_10_fu_464_p4 = {{gemm_stream_read_reg_902[178:158]}};

assign local_fc2_11_fu_477_p2 = ($signed(grp_fu_228_p4) + $signed(sext_ln294_5_fu_473_p1));

assign local_fc2_12_fu_483_p4 = {{gemm_stream_read_reg_902[208:188]}};

assign local_fc2_13_fu_496_p2 = ($signed(grp_fu_237_p4) + $signed(sext_ln294_6_fu_492_p1));

assign local_fc2_14_fu_502_p4 = {{gemm_stream_read_reg_902[238:218]}};

assign local_fc2_15_fu_515_p2 = ($signed(grp_fu_246_p4) + $signed(sext_ln294_7_fu_511_p1));

assign local_fc2_1_fu_382_p2 = ($signed(trunc_ln295_fu_379_p1) + $signed(sext_ln294_fu_375_p1));

assign local_fc2_2_fu_388_p4 = {{gemm_stream_read_reg_902[58:38]}};

assign local_fc2_3_fu_401_p2 = ($signed(grp_fu_192_p4) + $signed(sext_ln294_1_fu_397_p1));

assign local_fc2_4_fu_407_p4 = {{gemm_stream_read_reg_902[88:68]}};

assign local_fc2_5_fu_420_p2 = ($signed(grp_fu_201_p4) + $signed(sext_ln294_2_fu_416_p1));

assign local_fc2_6_fu_426_p4 = {{gemm_stream_read_reg_902[118:98]}};

assign local_fc2_7_fu_439_p2 = ($signed(grp_fu_210_p4) + $signed(sext_ln294_3_fu_435_p1));

assign local_fc2_8_fu_445_p4 = {{gemm_stream_read_reg_902[148:128]}};

assign local_fc2_9_fu_458_p2 = ($signed(grp_fu_219_p4) + $signed(sext_ln294_4_fu_454_p1));

assign local_fc2_fu_366_p4 = {{gemm_stream_read_reg_902[28:8]}};

assign local_o_10_fu_731_p4 = {{gemm_stream_read_reg_902[178:160]}};

assign local_o_11_fu_744_p2 = ($signed(grp_fu_228_p4) + $signed(sext_ln270_5_fu_740_p1));

assign local_o_12_fu_750_p4 = {{gemm_stream_read_reg_902[208:190]}};

assign local_o_13_fu_763_p2 = ($signed(grp_fu_237_p4) + $signed(sext_ln270_6_fu_759_p1));

assign local_o_14_fu_769_p4 = {{gemm_stream_read_reg_902[238:220]}};

assign local_o_15_fu_782_p2 = ($signed(grp_fu_246_p4) + $signed(sext_ln270_7_fu_778_p1));

assign local_o_1_fu_649_p2 = ($signed(trunc_ln271_fu_646_p1) + $signed(sext_ln270_fu_642_p1));

assign local_o_2_fu_655_p4 = {{gemm_stream_read_reg_902[58:40]}};

assign local_o_3_fu_668_p2 = ($signed(grp_fu_192_p4) + $signed(sext_ln270_1_fu_664_p1));

assign local_o_4_fu_674_p4 = {{gemm_stream_read_reg_902[88:70]}};

assign local_o_5_fu_687_p2 = ($signed(grp_fu_201_p4) + $signed(sext_ln270_2_fu_683_p1));

assign local_o_6_fu_693_p4 = {{gemm_stream_read_reg_902[118:100]}};

assign local_o_7_fu_706_p2 = ($signed(grp_fu_210_p4) + $signed(sext_ln270_3_fu_702_p1));

assign local_o_8_fu_712_p4 = {{gemm_stream_read_reg_902[148:130]}};

assign local_o_9_fu_725_p2 = ($signed(grp_fu_219_p4) + $signed(sext_ln270_4_fu_721_p1));

assign local_o_fu_633_p4 = {{gemm_stream_read_reg_902[28:10]}};

assign mlp1_stream_TDATA = {{{{{{{{local_fc1_15_reg_1066}, {local_fc1_13_reg_1061}}, {local_fc1_11_reg_1056}}, {local_fc1_9_reg_1051}}, {local_fc1_7_reg_1046}}, {local_fc1_5_reg_1041}}, {local_fc1_3_reg_1036}}, {local_fc1_1_reg_1031}};

assign sext_ln270_1_fu_664_p1 = $signed(local_o_2_fu_655_p4);

assign sext_ln270_2_fu_683_p1 = $signed(local_o_4_fu_674_p4);

assign sext_ln270_3_fu_702_p1 = $signed(local_o_6_fu_693_p4);

assign sext_ln270_4_fu_721_p1 = $signed(local_o_8_fu_712_p4);

assign sext_ln270_5_fu_740_p1 = $signed(local_o_10_fu_731_p4);

assign sext_ln270_6_fu_759_p1 = $signed(local_o_12_fu_750_p4);

assign sext_ln270_7_fu_778_p1 = $signed(local_o_14_fu_769_p4);

assign sext_ln270_fu_642_p1 = $signed(local_o_fu_633_p4);

assign sext_ln272_fu_846_p1 = $signed(local_o_1_reg_1071);

assign sext_ln274_1_fu_852_p1 = $signed(local_o_5_reg_1081);

assign sext_ln274_2_fu_855_p1 = $signed(local_o_7_reg_1086);

assign sext_ln274_3_fu_858_p1 = $signed(local_o_9_reg_1091);

assign sext_ln274_4_fu_861_p1 = $signed(local_o_11_reg_1096);

assign sext_ln274_5_fu_864_p1 = $signed(local_o_13_reg_1101);

assign sext_ln274_6_fu_886_p1 = $signed(tmp_fu_867_p9);

assign sext_ln274_fu_849_p1 = $signed(local_o_3_reg_1076);

assign sext_ln294_1_fu_397_p1 = $signed(local_fc2_2_fu_388_p4);

assign sext_ln294_2_fu_416_p1 = $signed(local_fc2_4_fu_407_p4);

assign sext_ln294_3_fu_435_p1 = $signed(local_fc2_6_fu_426_p4);

assign sext_ln294_4_fu_454_p1 = $signed(local_fc2_8_fu_445_p4);

assign sext_ln294_5_fu_473_p1 = $signed(local_fc2_10_fu_464_p4);

assign sext_ln294_6_fu_492_p1 = $signed(local_fc2_12_fu_483_p4);

assign sext_ln294_7_fu_511_p1 = $signed(local_fc2_14_fu_502_p4);

assign sext_ln294_fu_375_p1 = $signed(local_fc2_fu_366_p4);

assign sext_ln296_fu_788_p1 = $signed(local_fc2_1_reg_991);

assign sext_ln298_1_fu_794_p1 = $signed(local_fc2_5_reg_1001);

assign sext_ln298_2_fu_797_p1 = $signed(local_fc2_7_reg_1006);

assign sext_ln298_3_fu_800_p1 = $signed(local_fc2_9_reg_1011);

assign sext_ln298_4_fu_803_p1 = $signed(local_fc2_11_reg_1016);

assign sext_ln298_5_fu_806_p1 = $signed(local_fc2_13_reg_1021);

assign sext_ln298_6_fu_828_p1 = $signed(tmp_1_fu_809_p9);

assign sext_ln298_fu_791_p1 = $signed(local_fc2_3_reg_996);

assign tmp_1_fu_809_p9 = {{{{{{{{local_fc2_15_reg_1026}, {sext_ln298_5_fu_806_p1}}, {sext_ln298_4_fu_803_p1}}, {sext_ln298_3_fu_800_p1}}, {sext_ln298_2_fu_797_p1}}, {sext_ln298_1_fu_794_p1}}, {sext_ln298_fu_791_p1}}, {sext_ln296_fu_788_p1}};

assign tmp_fu_867_p9 = {{{{{{{{local_o_15_reg_1106}, {sext_ln274_5_fu_864_p1}}, {sext_ln274_4_fu_861_p1}}, {sext_ln274_3_fu_858_p1}}, {sext_ln274_2_fu_855_p1}}, {sext_ln274_1_fu_852_p1}}, {sext_ln274_fu_849_p1}}, {sext_ln272_fu_846_p1}};

assign trunc_ln262_fu_269_p1 = bias_stream_TDATA[27:0];

assign trunc_ln271_fu_646_p1 = bias_stream_read_reg_930[22:0];

assign trunc_ln295_fu_379_p1 = bias_stream_read_reg_930[22:0];

endmodule //DEMUX_demux_vit_encoder_Pipeline_VITIS_LOOP_259_7
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_llm_cls_Pipeline_VITIS_LOOP_199_3 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        empty_21,
        empty_22,
        vocabt,
        p_cast15,
        empty_23,
        empty,
        cls_idx_2_out_i,
        cls_idx_2_out_o,
        cls_idx_2_out_o_ap_vld,
        p_out_i,
        p_out_o,
        p_out_o_ap_vld,
        p_out1,
        p_out1_ap_vld
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
input  [22:0] empty_21;
input  [239:0] empty_22;
input  [12:0] vocabt;
input  [7:0] p_cast15;
input  [239:0] empty_23;
input  [255:0] empty;
input  [255:0] cls_idx_2_out_i;
output  [255:0] cls_idx_2_out_o;
output   cls_idx_2_out_o_ap_vld;
input  [255:0] p_out_i;
output  [255:0] p_out_o;
output   p_out_o_ap_vld;
output  [22:0] p_out1;
output   p_out1_ap_vld;

reg ap_idle;
reg[255:0] cls_idx_2_out_o;
reg cls_idx_2_out_o_ap_vld;
reg[255:0] p_out_o;
reg p_out_o_ap_vld;
reg p_out1_ap_vld;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
wire   [0:0] icmp_ln199_fu_136_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
wire   [239:0] p_cast15_cast_fu_119_p1;
reg   [239:0] p_cast15_cast_reg_355;
reg   [0:0] icmp_ln199_reg_360;
reg   [0:0] icmp_ln199_reg_360_pp0_iter1_reg;
wire   [2:0] trunc_ln201_fu_148_p1;
reg   [2:0] trunc_ln201_reg_364;
reg   [2:0] trunc_ln201_reg_364_pp0_iter1_reg;
wire   [8:0] sub_ln201_fu_174_p2;
reg   [8:0] sub_ln201_reg_369;
wire   [22:0] cls_val_fu_197_p1;
reg   [22:0] cls_val_reg_374;
wire   [255:0] select_ln203_fu_276_p3;
wire   [255:0] select_ln203_1_fu_284_p3;
reg   [3:0] cp_fu_68;
wire   [3:0] add_ln199_fu_142_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_cp_1;
reg   [22:0] empty_32_fu_72;
wire   [22:0] cls_val_1_fu_292_p3;
wire   [7:0] shl_ln_fu_152_p3;
wire   [3:0] shl_ln201_fu_164_p2;
wire   [8:0] zext_ln201_fu_160_p1;
wire   [8:0] zext_ln201_1_fu_170_p1;
wire  signed [63:0] sext_ln201_fu_185_p1;
wire   [239:0] zext_ln201_2_fu_188_p1;
wire   [239:0] lshr_ln201_fu_192_p2;
wire   [15:0] add_ln_fu_217_p3;
wire   [239:0] zext_ln205_fu_223_p1;
wire   [239:0] trunc_ln205_fu_237_p1;
wire   [239:0] xor_ln205_fu_232_p2;
wire   [239:0] and_ln205_1_fu_246_p2;
wire   [239:0] shl_ln205_fu_227_p2;
wire   [255:0] and_ln205_fu_241_p2;
wire   [15:0] tmp_fu_258_p4;
wire   [239:0] or_ln205_fu_252_p2;
wire   [0:0] icmp_ln203_fu_212_p2;
wire   [255:0] or_ln_fu_268_p3;
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
wire    ap_ST_iter2_fsm_state3_blk;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 cp_fu_68 = 4'd0;
//#0 empty_32_fu_72 = 23'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        end else if (((1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        if ((icmp_ln199_fu_136_p2 == 1'd0)) begin
            cp_fu_68 <= add_ln199_fu_142_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            cp_fu_68 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        empty_32_fu_72 <= empty_21;
    end else if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd0))) begin
        empty_32_fu_72 <= cls_val_1_fu_292_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln199_reg_360 <= icmp_ln199_fu_136_p2;
        p_cast15_cast_reg_355[7 : 0] <= p_cast15_cast_fu_119_p1[7 : 0];
        sub_ln201_reg_369[8 : 1] <= sub_ln201_fu_174_p2[8 : 1];
        trunc_ln201_reg_364 <= trunc_ln201_fu_148_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_iter1_fsm_state2)) begin
        cls_val_reg_374 <= cls_val_fu_197_p1;
        icmp_ln199_reg_360_pp0_iter1_reg <= icmp_ln199_reg_360;
        trunc_ln201_reg_364_pp0_iter1_reg <= trunc_ln201_reg_364;
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
    if (((icmp_ln199_fu_136_p2 == 1'd1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1))) begin
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
    if (((1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_cp_1 = 4'd0;
    end else begin
        ap_sig_allocacmp_cp_1 = cp_fu_68;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd0))) begin
        cls_idx_2_out_o = select_ln203_fu_276_p3;
    end else begin
        cls_idx_2_out_o = cls_idx_2_out_i;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd0))) begin
        cls_idx_2_out_o_ap_vld = 1'b1;
    end else begin
        cls_idx_2_out_o_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd1))) begin
        p_out1_ap_vld = 1'b1;
    end else begin
        p_out1_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd0))) begin
        p_out_o = select_ln203_1_fu_284_p3;
    end else begin
        p_out_o = p_out_i;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd0))) begin
        p_out_o_ap_vld = 1'b1;
    end else begin
        p_out_o_ap_vld = 1'b0;
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
            if ((1'b0 == ap_CS_iter1_fsm_state2)) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if (((1'b1 == ap_CS_iter1_fsm_state2) | ((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln199_reg_360_pp0_iter1_reg == 1'd1)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
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

assign add_ln199_fu_142_p2 = (ap_sig_allocacmp_cp_1 + 4'd1);

assign add_ln_fu_217_p3 = {{vocabt}, {trunc_ln201_reg_364_pp0_iter1_reg}};

assign and_ln205_1_fu_246_p2 = (xor_ln205_fu_232_p2 & trunc_ln205_fu_237_p1);

assign and_ln205_fu_241_p2 = (empty & cls_idx_2_out_i);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign cls_val_1_fu_292_p3 = ((icmp_ln203_fu_212_p2[0:0] == 1'b1) ? cls_val_reg_374 : empty_32_fu_72);

assign cls_val_fu_197_p1 = lshr_ln201_fu_192_p2[22:0];

assign icmp_ln199_fu_136_p2 = ((ap_sig_allocacmp_cp_1 == 4'd8) ? 1'b1 : 1'b0);

assign icmp_ln203_fu_212_p2 = (($signed(cls_val_reg_374) > $signed(empty_32_fu_72)) ? 1'b1 : 1'b0);

assign lshr_ln201_fu_192_p2 = empty_22 >> zext_ln201_2_fu_188_p1;

assign or_ln205_fu_252_p2 = (shl_ln205_fu_227_p2 | and_ln205_1_fu_246_p2);

assign or_ln_fu_268_p3 = {{tmp_fu_258_p4}, {or_ln205_fu_252_p2}};

assign p_cast15_cast_fu_119_p1 = p_cast15;

assign p_out1 = empty_32_fu_72;

assign select_ln203_1_fu_284_p3 = ((icmp_ln203_fu_212_p2[0:0] == 1'b1) ? or_ln_fu_268_p3 : p_out_i);

assign select_ln203_fu_276_p3 = ((icmp_ln203_fu_212_p2[0:0] == 1'b1) ? or_ln_fu_268_p3 : cls_idx_2_out_i);

assign sext_ln201_fu_185_p1 = $signed(sub_ln201_reg_369);

assign shl_ln201_fu_164_p2 = ap_sig_allocacmp_cp_1 << 4'd1;

assign shl_ln205_fu_227_p2 = zext_ln205_fu_223_p1 << p_cast15_cast_reg_355;

assign shl_ln_fu_152_p3 = {{trunc_ln201_fu_148_p1}, {5'd0}};

assign sub_ln201_fu_174_p2 = (zext_ln201_fu_160_p1 - zext_ln201_1_fu_170_p1);

assign tmp_fu_258_p4 = {{and_ln205_fu_241_p2[255:240]}};

assign trunc_ln201_fu_148_p1 = ap_sig_allocacmp_cp_1[2:0];

assign trunc_ln205_fu_237_p1 = cls_idx_2_out_i[239:0];

assign xor_ln205_fu_232_p2 = (empty_23 ^ 240'd1766847064778384329583297500742918515827483896875618958121606201292619775);

assign zext_ln201_1_fu_170_p1 = shl_ln201_fu_164_p2;

assign zext_ln201_2_fu_188_p1 = $unsigned(sext_ln201_fu_185_p1);

assign zext_ln201_fu_160_p1 = shl_ln_fu_152_p3;

assign zext_ln205_fu_223_p1 = add_ln_fu_217_p3;

always @ (posedge ap_clk) begin
    p_cast15_cast_reg_355[239:8] <= 232'b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
    sub_ln201_reg_369[0] <= 1'b0;
end

endmodule //DEMUX_demux_llm_cls_Pipeline_VITIS_LOOP_199_3
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="DEMUX_DEMUX,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xck26-sfvc784-2LV-c,HLS_INPUT_CLOCK=2.500000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=2.839000,HLS_SYN_LAT=5480445,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=5620,HLS_SYN_LUT=10330,HLS_VERSION=2023_2}" *)

module DEMUX (
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
        gemm_stream_TDATA,
        gemm_stream_TVALID,
        gemm_stream_TREADY,
        bias_stream_TDATA,
        bias_stream_TVALID,
        bias_stream_TREADY,
        qk_stream_TDATA,
        qk_stream_TVALID,
        qk_stream_TREADY,
        v_stream_TDATA,
        v_stream_TVALID,
        v_stream_TREADY,
        mlp1_stream_TDATA,
        mlp1_stream_TVALID,
        mlp1_stream_TREADY,
        od_fc2_stream_TDATA,
        od_fc2_stream_TVALID,
        od_fc2_stream_TREADY,
        cls_stream_TDATA,
        cls_stream_TVALID,
        cls_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 7'd1;
parameter    ap_ST_fsm_state2 = 7'd2;
parameter    ap_ST_fsm_state3 = 7'd4;
parameter    ap_ST_fsm_state4 = 7'd8;
parameter    ap_ST_fsm_state5 = 7'd16;
parameter    ap_ST_fsm_state6 = 7'd32;
parameter    ap_ST_fsm_state7 = 7'd64;

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
input  [239:0] gemm_stream_TDATA;
input   gemm_stream_TVALID;
output   gemm_stream_TREADY;
input  [223:0] bias_stream_TDATA;
input   bias_stream_TVALID;
output   bias_stream_TREADY;
output  [183:0] qk_stream_TDATA;
output   qk_stream_TVALID;
input   qk_stream_TREADY;
output  [183:0] v_stream_TDATA;
output   v_stream_TVALID;
input   v_stream_TREADY;
output  [223:0] mlp1_stream_TDATA;
output   mlp1_stream_TVALID;
input   mlp1_stream_TREADY;
output  [199:0] od_fc2_stream_TDATA;
output   od_fc2_stream_TVALID;
input   od_fc2_stream_TREADY;
output  [255:0] cls_stream_TDATA;
output   cls_stream_TVALID;
input   cls_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [6:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_block_state1;
wire   [0:0] mode_read_read_fu_104_p2;
wire   [2:0] decoded_chunk_pos_fu_153_p3;
reg   [2:0] decoded_chunk_pos_reg_239;
wire   [31:0] select_ln337_fu_161_p3;
reg   [31:0] select_ln337_reg_244;
wire   [0:0] and_ln59_fu_201_p2;
reg   [0:0] and_ln59_reg_255;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln64_fu_207_p2;
reg   [0:0] icmp_ln64_reg_259;
wire    grp_demux_llm_decoder_fu_110_ap_start;
wire    grp_demux_llm_decoder_fu_110_ap_done;
wire    grp_demux_llm_decoder_fu_110_ap_idle;
wire    grp_demux_llm_decoder_fu_110_ap_ready;
wire    grp_demux_llm_decoder_fu_110_gemm_stream_TREADY;
wire   [183:0] grp_demux_llm_decoder_fu_110_qk_stream_TDATA;
wire    grp_demux_llm_decoder_fu_110_qk_stream_TVALID;
wire    grp_demux_llm_decoder_fu_110_qk_stream_TREADY;
wire   [183:0] grp_demux_llm_decoder_fu_110_v_stream_TDATA;
wire    grp_demux_llm_decoder_fu_110_v_stream_TVALID;
wire    grp_demux_llm_decoder_fu_110_v_stream_TREADY;
wire   [223:0] grp_demux_llm_decoder_fu_110_mlp1_stream_TDATA;
wire    grp_demux_llm_decoder_fu_110_mlp1_stream_TVALID;
wire    grp_demux_llm_decoder_fu_110_mlp1_stream_TREADY;
wire   [199:0] grp_demux_llm_decoder_fu_110_od_fc2_stream_TDATA;
wire    grp_demux_llm_decoder_fu_110_od_fc2_stream_TVALID;
wire    grp_demux_llm_decoder_fu_110_od_fc2_stream_TREADY;
wire    grp_demux_vit_encoder_fu_125_ap_start;
wire    grp_demux_vit_encoder_fu_125_ap_done;
wire    grp_demux_vit_encoder_fu_125_ap_idle;
wire    grp_demux_vit_encoder_fu_125_ap_ready;
wire    grp_demux_vit_encoder_fu_125_gemm_stream_TREADY;
wire    grp_demux_vit_encoder_fu_125_bias_stream_TREADY;
wire   [183:0] grp_demux_vit_encoder_fu_125_qk_stream_TDATA;
wire    grp_demux_vit_encoder_fu_125_qk_stream_TVALID;
wire    grp_demux_vit_encoder_fu_125_qk_stream_TREADY;
wire   [183:0] grp_demux_vit_encoder_fu_125_v_stream_TDATA;
wire    grp_demux_vit_encoder_fu_125_v_stream_TVALID;
wire    grp_demux_vit_encoder_fu_125_v_stream_TREADY;
wire   [223:0] grp_demux_vit_encoder_fu_125_mlp1_stream_TDATA;
wire    grp_demux_vit_encoder_fu_125_mlp1_stream_TVALID;
wire    grp_demux_vit_encoder_fu_125_mlp1_stream_TREADY;
wire   [199:0] grp_demux_vit_encoder_fu_125_od_fc2_stream_TDATA;
wire    grp_demux_vit_encoder_fu_125_od_fc2_stream_TVALID;
wire    grp_demux_vit_encoder_fu_125_od_fc2_stream_TREADY;
wire    grp_demux_llm_cls_fu_141_ap_start;
wire    grp_demux_llm_cls_fu_141_ap_done;
wire    grp_demux_llm_cls_fu_141_ap_idle;
wire    grp_demux_llm_cls_fu_141_ap_ready;
wire    grp_demux_llm_cls_fu_141_gemm_stream_TREADY;
wire   [255:0] grp_demux_llm_cls_fu_141_cls_stream_TDATA;
wire    grp_demux_llm_cls_fu_141_cls_stream_TVALID;
wire    grp_demux_llm_cls_fu_141_cls_stream_TREADY;
reg    grp_demux_llm_decoder_fu_110_ap_start_reg;
wire   [0:0] icmp_ln337_fu_177_p2;
wire    ap_CS_fsm_state3;
reg    grp_demux_vit_encoder_fu_125_ap_start_reg;
wire    ap_CS_fsm_state6;
reg    grp_demux_llm_cls_fu_141_ap_start_reg;
wire    ap_CS_fsm_state4;
wire    ap_CS_fsm_state5;
reg   [31:0] l_1_fu_82;
wire   [31:0] l_fu_213_p2;
reg    ap_predicate_op60_call_state5;
reg    ap_block_state5_on_subcall_done;
wire   [2:0] trunc_ln303_fu_149_p1;
wire   [0:0] tmp_fu_182_p3;
wire   [0:0] icmp_ln59_fu_196_p2;
wire   [0:0] xor_ln59_fu_190_p2;
wire    ap_CS_fsm_state7;
wire    regslice_both_qk_stream_U_apdone_blk;
wire    regslice_both_v_stream_U_apdone_blk;
wire    regslice_both_mlp1_stream_U_apdone_blk;
wire    regslice_both_od_fc2_stream_U_apdone_blk;
wire    regslice_both_cls_stream_U_apdone_blk;
reg    ap_block_state7;
reg   [6:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
reg    ap_ST_fsm_state6_blk;
reg    ap_ST_fsm_state7_blk;
wire    regslice_both_gemm_stream_U_apdone_blk;
wire   [239:0] gemm_stream_TDATA_int_regslice;
wire    gemm_stream_TVALID_int_regslice;
reg    gemm_stream_TREADY_int_regslice;
wire    regslice_both_gemm_stream_U_ack_in;
wire    regslice_both_bias_stream_U_apdone_blk;
wire   [223:0] bias_stream_TDATA_int_regslice;
wire    bias_stream_TVALID_int_regslice;
reg    bias_stream_TREADY_int_regslice;
wire    regslice_both_bias_stream_U_ack_in;
reg   [183:0] qk_stream_TDATA_int_regslice;
reg    qk_stream_TVALID_int_regslice;
wire    qk_stream_TREADY_int_regslice;
wire    regslice_both_qk_stream_U_vld_out;
reg   [183:0] v_stream_TDATA_int_regslice;
reg    v_stream_TVALID_int_regslice;
wire    v_stream_TREADY_int_regslice;
wire    regslice_both_v_stream_U_vld_out;
reg   [223:0] mlp1_stream_TDATA_int_regslice;
reg    mlp1_stream_TVALID_int_regslice;
wire    mlp1_stream_TREADY_int_regslice;
wire    regslice_both_mlp1_stream_U_vld_out;
reg   [199:0] od_fc2_stream_TDATA_int_regslice;
reg    od_fc2_stream_TVALID_int_regslice;
wire    od_fc2_stream_TREADY_int_regslice;
wire    regslice_both_od_fc2_stream_U_vld_out;
wire    cls_stream_TREADY_int_regslice;
wire    regslice_both_cls_stream_U_vld_out;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 7'd1;
//#0 grp_demux_llm_decoder_fu_110_ap_start_reg = 1'b0;
//#0 grp_demux_vit_encoder_fu_125_ap_start_reg = 1'b0;
//#0 grp_demux_llm_cls_fu_141_ap_start_reg = 1'b0;
//#0 l_1_fu_82 = 32'd0;
end

DEMUX_demux_llm_decoder grp_demux_llm_decoder_fu_110(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_demux_llm_decoder_fu_110_ap_start),
    .ap_done(grp_demux_llm_decoder_fu_110_ap_done),
    .ap_idle(grp_demux_llm_decoder_fu_110_ap_idle),
    .ap_ready(grp_demux_llm_decoder_fu_110_ap_ready),
    .chunk_pos(decoded_chunk_pos_reg_239),
    .gemm_stream_TDATA(gemm_stream_TDATA_int_regslice),
    .gemm_stream_TVALID(gemm_stream_TVALID_int_regslice),
    .gemm_stream_TREADY(grp_demux_llm_decoder_fu_110_gemm_stream_TREADY),
    .qk_stream_TDATA(grp_demux_llm_decoder_fu_110_qk_stream_TDATA),
    .qk_stream_TVALID(grp_demux_llm_decoder_fu_110_qk_stream_TVALID),
    .qk_stream_TREADY(grp_demux_llm_decoder_fu_110_qk_stream_TREADY),
    .v_stream_TDATA(grp_demux_llm_decoder_fu_110_v_stream_TDATA),
    .v_stream_TVALID(grp_demux_llm_decoder_fu_110_v_stream_TVALID),
    .v_stream_TREADY(grp_demux_llm_decoder_fu_110_v_stream_TREADY),
    .mlp1_stream_TDATA(grp_demux_llm_decoder_fu_110_mlp1_stream_TDATA),
    .mlp1_stream_TVALID(grp_demux_llm_decoder_fu_110_mlp1_stream_TVALID),
    .mlp1_stream_TREADY(grp_demux_llm_decoder_fu_110_mlp1_stream_TREADY),
    .od_fc2_stream_TDATA(grp_demux_llm_decoder_fu_110_od_fc2_stream_TDATA),
    .od_fc2_stream_TVALID(grp_demux_llm_decoder_fu_110_od_fc2_stream_TVALID),
    .od_fc2_stream_TREADY(grp_demux_llm_decoder_fu_110_od_fc2_stream_TREADY)
);

DEMUX_demux_vit_encoder grp_demux_vit_encoder_fu_125(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_demux_vit_encoder_fu_125_ap_start),
    .ap_done(grp_demux_vit_encoder_fu_125_ap_done),
    .ap_idle(grp_demux_vit_encoder_fu_125_ap_idle),
    .ap_ready(grp_demux_vit_encoder_fu_125_ap_ready),
    .gemm_stream_TDATA(gemm_stream_TDATA_int_regslice),
    .gemm_stream_TVALID(gemm_stream_TVALID_int_regslice),
    .gemm_stream_TREADY(grp_demux_vit_encoder_fu_125_gemm_stream_TREADY),
    .bias_stream_TDATA(bias_stream_TDATA_int_regslice),
    .bias_stream_TVALID(bias_stream_TVALID_int_regslice),
    .bias_stream_TREADY(grp_demux_vit_encoder_fu_125_bias_stream_TREADY),
    .qk_stream_TDATA(grp_demux_vit_encoder_fu_125_qk_stream_TDATA),
    .qk_stream_TVALID(grp_demux_vit_encoder_fu_125_qk_stream_TVALID),
    .qk_stream_TREADY(grp_demux_vit_encoder_fu_125_qk_stream_TREADY),
    .v_stream_TDATA(grp_demux_vit_encoder_fu_125_v_stream_TDATA),
    .v_stream_TVALID(grp_demux_vit_encoder_fu_125_v_stream_TVALID),
    .v_stream_TREADY(grp_demux_vit_encoder_fu_125_v_stream_TREADY),
    .mlp1_stream_TDATA(grp_demux_vit_encoder_fu_125_mlp1_stream_TDATA),
    .mlp1_stream_TVALID(grp_demux_vit_encoder_fu_125_mlp1_stream_TVALID),
    .mlp1_stream_TREADY(grp_demux_vit_encoder_fu_125_mlp1_stream_TREADY),
    .od_fc2_stream_TDATA(grp_demux_vit_encoder_fu_125_od_fc2_stream_TDATA),
    .od_fc2_stream_TVALID(grp_demux_vit_encoder_fu_125_od_fc2_stream_TVALID),
    .od_fc2_stream_TREADY(grp_demux_vit_encoder_fu_125_od_fc2_stream_TREADY)
);

DEMUX_demux_llm_cls grp_demux_llm_cls_fu_141(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_demux_llm_cls_fu_141_ap_start),
    .ap_done(grp_demux_llm_cls_fu_141_ap_done),
    .ap_idle(grp_demux_llm_cls_fu_141_ap_idle),
    .ap_ready(grp_demux_llm_cls_fu_141_ap_ready),
    .gemm_stream_TDATA(gemm_stream_TDATA_int_regslice),
    .gemm_stream_TVALID(gemm_stream_TVALID_int_regslice),
    .gemm_stream_TREADY(grp_demux_llm_cls_fu_141_gemm_stream_TREADY),
    .cls_stream_TDATA(grp_demux_llm_cls_fu_141_cls_stream_TDATA),
    .cls_stream_TVALID(grp_demux_llm_cls_fu_141_cls_stream_TVALID),
    .cls_stream_TREADY(grp_demux_llm_cls_fu_141_cls_stream_TREADY)
);

DEMUX_regslice_both #(
    .DataWidth( 240 ))
regslice_both_gemm_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(gemm_stream_TDATA),
    .vld_in(gemm_stream_TVALID),
    .ack_in(regslice_both_gemm_stream_U_ack_in),
    .data_out(gemm_stream_TDATA_int_regslice),
    .vld_out(gemm_stream_TVALID_int_regslice),
    .ack_out(gemm_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_gemm_stream_U_apdone_blk)
);

DEMUX_regslice_both #(
    .DataWidth( 224 ))
regslice_both_bias_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(bias_stream_TDATA),
    .vld_in(bias_stream_TVALID),
    .ack_in(regslice_both_bias_stream_U_ack_in),
    .data_out(bias_stream_TDATA_int_regslice),
    .vld_out(bias_stream_TVALID_int_regslice),
    .ack_out(bias_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_bias_stream_U_apdone_blk)
);

DEMUX_regslice_both #(
    .DataWidth( 184 ))
regslice_both_qk_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(qk_stream_TDATA_int_regslice),
    .vld_in(qk_stream_TVALID_int_regslice),
    .ack_in(qk_stream_TREADY_int_regslice),
    .data_out(qk_stream_TDATA),
    .vld_out(regslice_both_qk_stream_U_vld_out),
    .ack_out(qk_stream_TREADY),
    .apdone_blk(regslice_both_qk_stream_U_apdone_blk)
);

DEMUX_regslice_both #(
    .DataWidth( 184 ))
regslice_both_v_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(v_stream_TDATA_int_regslice),
    .vld_in(v_stream_TVALID_int_regslice),
    .ack_in(v_stream_TREADY_int_regslice),
    .data_out(v_stream_TDATA),
    .vld_out(regslice_both_v_stream_U_vld_out),
    .ack_out(v_stream_TREADY),
    .apdone_blk(regslice_both_v_stream_U_apdone_blk)
);

DEMUX_regslice_both #(
    .DataWidth( 224 ))
regslice_both_mlp1_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(mlp1_stream_TDATA_int_regslice),
    .vld_in(mlp1_stream_TVALID_int_regslice),
    .ack_in(mlp1_stream_TREADY_int_regslice),
    .data_out(mlp1_stream_TDATA),
    .vld_out(regslice_both_mlp1_stream_U_vld_out),
    .ack_out(mlp1_stream_TREADY),
    .apdone_blk(regslice_both_mlp1_stream_U_apdone_blk)
);

DEMUX_regslice_both #(
    .DataWidth( 200 ))
regslice_both_od_fc2_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(od_fc2_stream_TDATA_int_regslice),
    .vld_in(od_fc2_stream_TVALID_int_regslice),
    .ack_in(od_fc2_stream_TREADY_int_regslice),
    .data_out(od_fc2_stream_TDATA),
    .vld_out(regslice_both_od_fc2_stream_U_vld_out),
    .ack_out(od_fc2_stream_TREADY),
    .apdone_blk(regslice_both_od_fc2_stream_U_apdone_blk)
);

DEMUX_regslice_both #(
    .DataWidth( 256 ))
regslice_both_cls_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_demux_llm_cls_fu_141_cls_stream_TDATA),
    .vld_in(grp_demux_llm_cls_fu_141_cls_stream_TVALID),
    .ack_in(cls_stream_TREADY_int_regslice),
    .data_out(cls_stream_TDATA),
    .vld_out(regslice_both_cls_stream_U_vld_out),
    .ack_out(cls_stream_TREADY),
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
        end else if (((1'b0 == ap_block_state7) & (1'b1 == ap_CS_fsm_state7))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_demux_llm_cls_fu_141_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state4)) begin
            grp_demux_llm_cls_fu_141_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_cls_fu_141_ap_ready == 1'b1)) begin
            grp_demux_llm_cls_fu_141_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_demux_llm_decoder_fu_110_ap_start_reg <= 1'b0;
    end else begin
        if (((icmp_ln64_fu_207_p2 == 1'd0) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state2) & (1'd1 == and_ln59_fu_201_p2) & (icmp_ln337_fu_177_p2 == 1'd1))) begin
            grp_demux_llm_decoder_fu_110_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_decoder_fu_110_ap_ready == 1'b1)) begin
            grp_demux_llm_decoder_fu_110_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_demux_vit_encoder_fu_125_ap_start_reg <= 1'b0;
    end else begin
        if (((mode_read_read_fu_104_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2) & (1'd1 == and_ln59_fu_201_p2) & (icmp_ln337_fu_177_p2 == 1'd1))) begin
            grp_demux_vit_encoder_fu_125_ap_start_reg <= 1'b1;
        end else if ((grp_demux_vit_encoder_fu_125_ap_ready == 1'b1)) begin
            grp_demux_vit_encoder_fu_125_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        l_1_fu_82 <= l_begin;
    end else if (((1'b0 == ap_block_state5_on_subcall_done) & (1'b1 == ap_CS_fsm_state5))) begin
        l_1_fu_82 <= l_fu_213_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        and_ln59_reg_255 <= and_ln59_fu_201_p2;
        icmp_ln64_reg_259 <= icmp_ln64_fu_207_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
        decoded_chunk_pos_reg_239 <= decoded_chunk_pos_fu_153_p3;
        select_ln337_reg_244[0] <= select_ln337_fu_161_p3[0];
select_ln337_reg_244[3 : 2] <= select_ln337_fu_161_p3[3 : 2];
select_ln337_reg_244[5] <= select_ln337_fu_161_p3[5];
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
    if ((grp_demux_llm_decoder_fu_110_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state5_on_subcall_done)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((grp_demux_vit_encoder_fu_125_ap_done == 1'b0)) begin
        ap_ST_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state7)) begin
        ap_ST_fsm_state7_blk = 1'b1;
    end else begin
        ap_ST_fsm_state7_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state7) & (1'b1 == ap_CS_fsm_state7))) begin
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
    if (((1'b0 == ap_block_state7) & (1'b1 == ap_CS_fsm_state7))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        bias_stream_TREADY_int_regslice = grp_demux_vit_encoder_fu_125_bias_stream_TREADY;
    end else begin
        bias_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln64_reg_259 == 1'd1) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state5) & (1'd1 == and_ln59_reg_255))) begin
        gemm_stream_TREADY_int_regslice = grp_demux_llm_cls_fu_141_gemm_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        gemm_stream_TREADY_int_regslice = grp_demux_vit_encoder_fu_125_gemm_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        gemm_stream_TREADY_int_regslice = grp_demux_llm_decoder_fu_110_gemm_stream_TREADY;
    end else begin
        gemm_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_demux_vit_encoder_fu_125_mlp1_stream_TVALID == 1'b1))) begin
        mlp1_stream_TDATA_int_regslice = grp_demux_vit_encoder_fu_125_mlp1_stream_TDATA;
    end else if (((grp_demux_llm_decoder_fu_110_mlp1_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
        mlp1_stream_TDATA_int_regslice = grp_demux_llm_decoder_fu_110_mlp1_stream_TDATA;
    end else begin
        mlp1_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        mlp1_stream_TVALID_int_regslice = grp_demux_vit_encoder_fu_125_mlp1_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        mlp1_stream_TVALID_int_regslice = grp_demux_llm_decoder_fu_110_mlp1_stream_TVALID;
    end else begin
        mlp1_stream_TVALID_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_demux_vit_encoder_fu_125_od_fc2_stream_TVALID == 1'b1))) begin
        od_fc2_stream_TDATA_int_regslice = grp_demux_vit_encoder_fu_125_od_fc2_stream_TDATA;
    end else if (((1'b1 == ap_CS_fsm_state3) & (grp_demux_llm_decoder_fu_110_od_fc2_stream_TVALID == 1'b1))) begin
        od_fc2_stream_TDATA_int_regslice = grp_demux_llm_decoder_fu_110_od_fc2_stream_TDATA;
    end else begin
        od_fc2_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        od_fc2_stream_TVALID_int_regslice = grp_demux_vit_encoder_fu_125_od_fc2_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        od_fc2_stream_TVALID_int_regslice = grp_demux_llm_decoder_fu_110_od_fc2_stream_TVALID;
    end else begin
        od_fc2_stream_TVALID_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_demux_vit_encoder_fu_125_qk_stream_TVALID == 1'b1))) begin
        qk_stream_TDATA_int_regslice = grp_demux_vit_encoder_fu_125_qk_stream_TDATA;
    end else if (((grp_demux_llm_decoder_fu_110_qk_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
        qk_stream_TDATA_int_regslice = grp_demux_llm_decoder_fu_110_qk_stream_TDATA;
    end else begin
        qk_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        qk_stream_TVALID_int_regslice = grp_demux_vit_encoder_fu_125_qk_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        qk_stream_TVALID_int_regslice = grp_demux_llm_decoder_fu_110_qk_stream_TVALID;
    end else begin
        qk_stream_TVALID_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_demux_vit_encoder_fu_125_v_stream_TVALID == 1'b1))) begin
        v_stream_TDATA_int_regslice = grp_demux_vit_encoder_fu_125_v_stream_TDATA;
    end else if (((grp_demux_llm_decoder_fu_110_v_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
        v_stream_TDATA_int_regslice = grp_demux_llm_decoder_fu_110_v_stream_TDATA;
    end else begin
        v_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        v_stream_TVALID_int_regslice = grp_demux_vit_encoder_fu_125_v_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        v_stream_TVALID_int_regslice = grp_demux_llm_decoder_fu_110_v_stream_TVALID;
    end else begin
        v_stream_TVALID_int_regslice = 1'b0;
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
            if (((icmp_ln64_fu_207_p2 == 1'd1) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state2) & (1'd1 == and_ln59_fu_201_p2) & (icmp_ln337_fu_177_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else if (((icmp_ln64_fu_207_p2 == 1'd0) & (mode == 1'd0) & (1'b1 == ap_CS_fsm_state2) & (1'd1 == and_ln59_fu_201_p2) & (icmp_ln337_fu_177_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else if (((mode_read_read_fu_104_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2) & (1'd1 == and_ln59_fu_201_p2) & (icmp_ln337_fu_177_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end else if (((1'b1 == ap_CS_fsm_state2) & (1'd0 == and_ln59_fu_201_p2) & (icmp_ln337_fu_177_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((grp_demux_llm_decoder_fu_110_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((1'b0 == ap_block_state5_on_subcall_done) & (1'b1 == ap_CS_fsm_state5))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_CS_fsm_state6) & (grp_demux_vit_encoder_fu_125_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state7 : begin
            if (((1'b0 == ap_block_state7) & (1'b1 == ap_CS_fsm_state7))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign and_ln59_fu_201_p2 = (xor_ln59_fu_190_p2 & icmp_ln59_fu_196_p2);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

always @ (*) begin
    ap_block_state1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0));
end

always @ (*) begin
    ap_block_state5_on_subcall_done = ((ap_predicate_op60_call_state5 == 1'b1) & (grp_demux_llm_cls_fu_141_ap_done == 1'b0));
end

always @ (*) begin
    ap_block_state7 = ((regslice_both_cls_stream_U_apdone_blk == 1'b1) | (regslice_both_od_fc2_stream_U_apdone_blk == 1'b1) | (regslice_both_mlp1_stream_U_apdone_blk == 1'b1) | (regslice_both_v_stream_U_apdone_blk == 1'b1) | (regslice_both_qk_stream_U_apdone_blk == 1'b1));
end

always @ (*) begin
    ap_predicate_op60_call_state5 = ((icmp_ln64_reg_259 == 1'd1) & (mode == 1'd0) & (1'd1 == and_ln59_reg_255));
end

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign bias_stream_TREADY = regslice_both_bias_stream_U_ack_in;

assign cls_stream_TVALID = regslice_both_cls_stream_U_vld_out;

assign decoded_chunk_pos_fu_153_p3 = ((mode[0:0] == 1'b1) ? 3'd0 : trunc_ln303_fu_149_p1);

assign gemm_stream_TREADY = regslice_both_gemm_stream_U_ack_in;

assign grp_demux_llm_cls_fu_141_ap_start = grp_demux_llm_cls_fu_141_ap_start_reg;

assign grp_demux_llm_cls_fu_141_cls_stream_TREADY = (cls_stream_TREADY_int_regslice & ap_CS_fsm_state5);

assign grp_demux_llm_decoder_fu_110_ap_start = grp_demux_llm_decoder_fu_110_ap_start_reg;

assign grp_demux_llm_decoder_fu_110_mlp1_stream_TREADY = (mlp1_stream_TREADY_int_regslice & ap_CS_fsm_state3);

assign grp_demux_llm_decoder_fu_110_od_fc2_stream_TREADY = (od_fc2_stream_TREADY_int_regslice & ap_CS_fsm_state3);

assign grp_demux_llm_decoder_fu_110_qk_stream_TREADY = (qk_stream_TREADY_int_regslice & ap_CS_fsm_state3);

assign grp_demux_llm_decoder_fu_110_v_stream_TREADY = (v_stream_TREADY_int_regslice & ap_CS_fsm_state3);

assign grp_demux_vit_encoder_fu_125_ap_start = grp_demux_vit_encoder_fu_125_ap_start_reg;

assign grp_demux_vit_encoder_fu_125_mlp1_stream_TREADY = (mlp1_stream_TREADY_int_regslice & ap_CS_fsm_state6);

assign grp_demux_vit_encoder_fu_125_od_fc2_stream_TREADY = (od_fc2_stream_TREADY_int_regslice & ap_CS_fsm_state6);

assign grp_demux_vit_encoder_fu_125_qk_stream_TREADY = (qk_stream_TREADY_int_regslice & ap_CS_fsm_state6);

assign grp_demux_vit_encoder_fu_125_v_stream_TREADY = (v_stream_TREADY_int_regslice & ap_CS_fsm_state6);

assign icmp_ln337_fu_177_p2 = (($signed(l_1_fu_82) < $signed(l_close)) ? 1'b1 : 1'b0);

assign icmp_ln59_fu_196_p2 = (($signed(l_1_fu_82) < $signed(select_ln337_reg_244)) ? 1'b1 : 1'b0);

assign icmp_ln64_fu_207_p2 = ((l_1_fu_82 == 32'd32) ? 1'b1 : 1'b0);

assign l_fu_213_p2 = (l_1_fu_82 + 32'd1);

assign mlp1_stream_TVALID = regslice_both_mlp1_stream_U_vld_out;

assign mode_read_read_fu_104_p2 = mode;

assign od_fc2_stream_TVALID = regslice_both_od_fc2_stream_U_vld_out;

assign qk_stream_TVALID = regslice_both_qk_stream_U_vld_out;

assign select_ln337_fu_161_p3 = ((mode[0:0] == 1'b1) ? 32'd12 : 32'd33);

assign tmp_fu_182_p3 = l_1_fu_82[32'd31];

assign trunc_ln303_fu_149_p1 = pos_r[2:0];

assign v_stream_TVALID = regslice_both_v_stream_U_vld_out;

assign xor_ln59_fu_190_p2 = (tmp_fu_182_p3 ^ 1'd1);

always @ (posedge ap_clk) begin
    select_ln337_reg_244[1] <= 1'b0;
    select_ln337_reg_244[4:4] <= 1'b0;
    select_ln337_reg_244[31:6] <= 26'b00000000000000000000000000;
end

endmodule //DEMUX
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TVALID,
        v_stream_TREADY,
        gemm_stream_TDATA,
        gemm_stream_TREADY,
        chunk_pos_cast,
        v_stream_TDATA,
        v_stream_TVALID
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
input   gemm_stream_TVALID;
input   v_stream_TREADY;
input  [239:0] gemm_stream_TDATA;
output   gemm_stream_TREADY;
input  [2:0] chunk_pos_cast;
output  [183:0] v_stream_TDATA;
output   v_stream_TVALID;

reg ap_idle;
reg gemm_stream_TREADY;
reg v_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln111_fu_130_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln111_reg_355;
wire   [0:0] icmp_ln111_reg_355_pp0_iter0_reg;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gemm_stream_TDATA_blk_n;
reg    v_stream_TDATA_blk_n;
wire   [3:0] chunk_pos_cast_cast_fu_113_p1;
reg   [3:0] chunk_pos_cast_cast_reg_350;
wire   [3:0] select_ln111_fu_151_p3;
reg   [3:0] select_ln111_reg_359;
reg   [22:0] local_v_reg_364;
reg   [22:0] local_v_2_reg_369;
reg   [22:0] local_v_4_reg_374;
reg   [22:0] local_v_6_reg_379;
reg   [22:0] local_v_8_reg_384;
reg   [22:0] local_v_10_reg_389;
reg   [22:0] local_v_12_reg_394;
reg   [22:0] local_v_14_reg_399;
reg   [3:0] t_fu_86;
wire   [3:0] t_1_fu_239_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_t_load;
reg   [6:0] indvar_flatten6_fu_90;
wire   [6:0] add_ln111_fu_136_p2;
reg   [6:0] ap_sig_allocacmp_indvar_flatten6_load;
wire   [0:0] icmp_ln112_fu_145_p2;
wire   [0:0] cmp42_fu_255_p2;
wire   [22:0] local_v_15_fu_308_p3;
wire   [22:0] local_v_13_fu_301_p3;
wire   [22:0] local_v_11_fu_294_p3;
wire   [22:0] local_v_9_fu_287_p3;
wire   [22:0] local_v_7_fu_280_p3;
wire   [22:0] local_v_5_fu_273_p3;
wire   [22:0] local_v_3_fu_266_p3;
wire   [22:0] local_v_1_fu_259_p3;
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
//#0 t_fu_86 = 4'd0;
//#0 indvar_flatten6_fu_90 = 7'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        if ((icmp_ln111_fu_130_p2 == 1'd0)) begin
            indvar_flatten6_fu_90 <= add_ln111_fu_136_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten6_fu_90 <= 7'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_63)) begin
        if ((icmp_ln111_fu_130_p2 == 1'd0)) begin
            t_fu_86 <= t_1_fu_239_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            t_fu_86 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        chunk_pos_cast_cast_reg_350[2 : 0] <= chunk_pos_cast_cast_fu_113_p1[2 : 0];
        icmp_ln111_reg_355 <= icmp_ln111_fu_130_p2;
        local_v_10_reg_389 <= {{gemm_stream_TDATA[176:154]}};
        local_v_12_reg_394 <= {{gemm_stream_TDATA[206:184]}};
        local_v_14_reg_399 <= {{gemm_stream_TDATA[236:214]}};
        local_v_2_reg_369 <= {{gemm_stream_TDATA[56:34]}};
        local_v_4_reg_374 <= {{gemm_stream_TDATA[86:64]}};
        local_v_6_reg_379 <= {{gemm_stream_TDATA[116:94]}};
        local_v_8_reg_384 <= {{gemm_stream_TDATA[146:124]}};
        local_v_reg_364 <= {{gemm_stream_TDATA[26:4]}};
        select_ln111_reg_359 <= select_ln111_fu_151_p3;
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln111_fu_130_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten6_load = 7'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten6_load = indvar_flatten6_fu_90;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_t_load = 4'd0;
    end else begin
        ap_sig_allocacmp_t_load = t_fu_86;
    end
end

always @ (*) begin
    if (((icmp_ln111_fu_130_p2 == 1'd0) & (ap_start_int == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln111_fu_130_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln111_reg_355 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        v_stream_TDATA_blk_n = v_stream_TREADY;
    end else begin
        v_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln111_reg_355 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        v_stream_TVALID = 1'b1;
    end else begin
        v_stream_TVALID = 1'b0;
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
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln111_reg_355_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
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

assign add_ln111_fu_136_p2 = (ap_sig_allocacmp_indvar_flatten6_load + 7'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln111_fu_130_p2 == 1'd0) & (gemm_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln111_reg_355 == 1'd0) & (v_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((icmp_ln111_reg_355 == 1'd0) & (v_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_63 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign chunk_pos_cast_cast_fu_113_p1 = chunk_pos_cast;

assign cmp42_fu_255_p2 = ((select_ln111_reg_359 > chunk_pos_cast_cast_reg_350) ? 1'b1 : 1'b0);

assign icmp_ln111_fu_130_p2 = ((ap_sig_allocacmp_indvar_flatten6_load == 7'd64) ? 1'b1 : 1'b0);

assign icmp_ln111_reg_355_pp0_iter0_reg = icmp_ln111_reg_355;

assign icmp_ln112_fu_145_p2 = ((ap_sig_allocacmp_t_load == 4'd8) ? 1'b1 : 1'b0);

assign local_v_11_fu_294_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_10_reg_389);

assign local_v_13_fu_301_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_12_reg_394);

assign local_v_15_fu_308_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_14_reg_399);

assign local_v_1_fu_259_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_reg_364);

assign local_v_3_fu_266_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_2_reg_369);

assign local_v_5_fu_273_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_4_reg_374);

assign local_v_7_fu_280_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_6_reg_379);

assign local_v_9_fu_287_p3 = ((cmp42_fu_255_p2[0:0] == 1'b1) ? 23'd0 : local_v_8_reg_384);

assign select_ln111_fu_151_p3 = ((icmp_ln112_fu_145_p2[0:0] == 1'b1) ? 4'd0 : ap_sig_allocacmp_t_load);

assign t_1_fu_239_p2 = (select_ln111_fu_151_p3 + 4'd1);

assign v_stream_TDATA = {{{{{{{{local_v_15_fu_308_p3}, {local_v_13_fu_301_p3}}, {local_v_11_fu_294_p3}}, {local_v_9_fu_287_p3}}, {local_v_7_fu_280_p3}}, {local_v_5_fu_273_p3}}, {local_v_3_fu_266_p3}}, {local_v_1_fu_259_p3}};

always @ (posedge ap_clk) begin
    chunk_pos_cast_cast_reg_350[3] <= 1'b0;
end

endmodule //DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_111_5_VITIS_LOOP_112_6
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TVALID,
        qk_stream_TREADY,
        gemm_stream_TDATA,
        gemm_stream_TREADY,
        qk_stream_TDATA,
        qk_stream_TVALID
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
input   gemm_stream_TVALID;
input   qk_stream_TREADY;
input  [239:0] gemm_stream_TDATA;
output   gemm_stream_TREADY;
output  [183:0] qk_stream_TDATA;
output   qk_stream_TVALID;

reg ap_idle;
reg gemm_stream_TREADY;
reg qk_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln96_fu_99_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln96_reg_216;
wire   [0:0] icmp_ln96_reg_216_pp0_iter0_reg;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gemm_stream_TDATA_blk_n;
reg    qk_stream_TDATA_blk_n;
reg   [22:0] trunc_ln_reg_220;
reg   [22:0] trunc_ln104_1_reg_225;
reg   [22:0] trunc_ln104_2_reg_230;
reg   [22:0] trunc_ln104_3_reg_235;
reg   [22:0] trunc_ln104_4_reg_240;
reg   [22:0] trunc_ln104_5_reg_245;
reg   [22:0] trunc_ln104_6_reg_250;
reg   [22:0] trunc_ln104_7_reg_255;
reg   [6:0] indvar_flatten_fu_74;
wire   [6:0] add_ln96_fu_105_p2;
wire    ap_loop_init;
reg   [6:0] ap_sig_allocacmp_indvar_flatten_load;
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
//#0 indvar_flatten_fu_74 = 7'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        if ((icmp_ln96_fu_99_p2 == 1'd0)) begin
            indvar_flatten_fu_74 <= add_ln96_fu_105_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_74 <= 7'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln96_reg_216 <= icmp_ln96_fu_99_p2;
        trunc_ln104_1_reg_225 <= {{gemm_stream_TDATA[55:33]}};
        trunc_ln104_2_reg_230 <= {{gemm_stream_TDATA[85:63]}};
        trunc_ln104_3_reg_235 <= {{gemm_stream_TDATA[115:93]}};
        trunc_ln104_4_reg_240 <= {{gemm_stream_TDATA[145:123]}};
        trunc_ln104_5_reg_245 <= {{gemm_stream_TDATA[175:153]}};
        trunc_ln104_6_reg_250 <= {{gemm_stream_TDATA[205:183]}};
        trunc_ln104_7_reg_255 <= {{gemm_stream_TDATA[235:213]}};
        trunc_ln_reg_220 <= {{gemm_stream_TDATA[25:3]}};
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln96_fu_99_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 7'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_74;
    end
end

always @ (*) begin
    if (((icmp_ln96_fu_99_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln96_fu_99_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln96_reg_216 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        qk_stream_TDATA_blk_n = qk_stream_TREADY;
    end else begin
        qk_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln96_reg_216 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        qk_stream_TVALID = 1'b1;
    end else begin
        qk_stream_TVALID = 1'b0;
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
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln96_reg_216_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
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

assign add_ln96_fu_105_p2 = (ap_sig_allocacmp_indvar_flatten_load + 7'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln96_fu_99_p2 == 1'd0) & (gemm_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln96_reg_216 == 1'd0) & (qk_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((icmp_ln96_reg_216 == 1'd0) & (qk_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_63 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln96_fu_99_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 7'd64) ? 1'b1 : 1'b0);

assign icmp_ln96_reg_216_pp0_iter0_reg = icmp_ln96_reg_216;

assign qk_stream_TDATA = {{{{{{{{trunc_ln104_7_reg_255}, {trunc_ln104_6_reg_250}}, {trunc_ln104_5_reg_245}}, {trunc_ln104_4_reg_240}}, {trunc_ln104_3_reg_235}}, {trunc_ln104_2_reg_230}}, {trunc_ln104_1_reg_225}}, {trunc_ln_reg_220}};

endmodule //DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_96_2_VITIS_LOOP_97_3
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module DEMUX_demux_llm_cls (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TDATA,
        gemm_stream_TVALID,
        gemm_stream_TREADY,
        cls_stream_TDATA,
        cls_stream_TVALID,
        cls_stream_TREADY
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
input  [239:0] gemm_stream_TDATA;
input   gemm_stream_TVALID;
output   gemm_stream_TREADY;
output  [255:0] cls_stream_TDATA;
output   cls_stream_TVALID;
input   cls_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg gemm_stream_TREADY;
reg cls_stream_TVALID;

(* fsm_encoding = "none" *) reg   [7:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    gemm_stream_TDATA_blk_n;
wire    ap_CS_fsm_state4;
reg    cls_stream_TDATA_blk_n;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln195_fu_162_p2;
reg   [12:0] vocabt_1_reg_320;
wire   [12:0] add_ln195_fu_168_p2;
reg   [12:0] add_ln195_reg_328;
wire   [7:0] add_ln196_1_fu_178_p2;
reg   [7:0] add_ln196_1_reg_333;
wire    ap_CS_fsm_state3;
wire   [3:0] add_ln196_fu_190_p2;
reg   [3:0] add_ln196_reg_341;
wire   [7:0] tmp_s_fu_203_p3;
reg   [7:0] tmp_s_reg_349;
wire   [183:0] zext_ln203_fu_211_p1;
reg   [183:0] zext_ln203_reg_354;
wire   [22:0] trunc_ln203_fu_221_p1;
reg   [22:0] trunc_ln203_reg_360;
wire   [255:0] empty_33_fu_229_p2;
reg   [255:0] empty_33_reg_365;
wire   [239:0] empty_34_fu_235_p1;
reg   [239:0] empty_34_reg_370;
reg   [239:0] gemm_stream_read_reg_375;
wire   [255:0] empty_35_fu_243_p2;
reg   [255:0] empty_35_reg_380;
wire   [183:0] or_ln203_fu_277_p2;
reg   [183:0] or_ln203_reg_385;
wire    ap_CS_fsm_state6;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_done;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_idle;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_ready;
wire   [255:0] grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_cls_idx_2_out_o;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_cls_idx_2_out_o_ap_vld;
wire   [255:0] grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out_o;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out_o_ap_vld;
wire   [22:0] grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out1;
wire    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out1_ap_vld;
reg   [3:0] t_reg_103;
reg    ap_block_state2;
reg    ap_block_state2_io;
wire    ap_CS_fsm_state7;
reg   [7:0] phi_mul_reg_114;
reg    grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start_reg;
wire    ap_CS_fsm_state5;
reg   [255:0] cls_idx_0_fu_78;
reg   [255:0] p_0_fu_74;
reg   [12:0] vocabt_fu_70;
wire   [0:0] icmp_ln196_fu_184_p2;
reg   [183:0] cls_max_0_fu_82;
wire   [2:0] empty_fu_199_p1;
wire   [183:0] lshr_ln203_fu_215_p2;
wire   [255:0] p_cast17_fu_225_p1;
wire   [183:0] zext_ln203_1_fu_257_p1;
wire   [183:0] shl_ln203_fu_252_p2;
wire   [183:0] xor_ln203_fu_266_p2;
wire   [183:0] shl_ln203_1_fu_261_p2;
wire   [183:0] and_ln203_fu_272_p2;
wire    ap_CS_fsm_state8;
reg   [7:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 8'd1;
//#0 grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start_reg = 1'b0;
//#0 cls_idx_0_fu_78 = 256'd0;
//#0 p_0_fu_74 = 256'd0;
//#0 vocabt_fu_70 = 13'd0;
//#0 cls_max_0_fu_82 = 184'd0;
end

DEMUX_demux_llm_cls_Pipeline_VITIS_LOOP_199_3 grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start),
    .ap_done(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_done),
    .ap_idle(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_idle),
    .ap_ready(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_ready),
    .empty_21(trunc_ln203_reg_360),
    .empty_22(gemm_stream_read_reg_375),
    .vocabt(vocabt_1_reg_320),
    .p_cast15(tmp_s_reg_349),
    .empty_23(empty_34_reg_370),
    .empty(empty_35_reg_380),
    .cls_idx_2_out_i(cls_idx_0_fu_78),
    .cls_idx_2_out_o(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_cls_idx_2_out_o),
    .cls_idx_2_out_o_ap_vld(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_cls_idx_2_out_o_ap_vld),
    .p_out_i(p_0_fu_74),
    .p_out_o(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out_o),
    .p_out_o_ap_vld(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out_o_ap_vld),
    .p_out1(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out1),
    .p_out1_ap_vld(grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out1_ap_vld)
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
        grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start_reg <= 1'b0;
    end else begin
        if (((gemm_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
            grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start_reg <= 1'b1;
        end else if ((grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_ready == 1'b1)) begin
            grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        cls_idx_0_fu_78 <= 256'd0;
    end else if (((grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_cls_idx_2_out_o_ap_vld == 1'b1) & (1'b1 == ap_CS_fsm_state5))) begin
        cls_idx_0_fu_78 <= grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_cls_idx_2_out_o;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        cls_max_0_fu_82 <= 184'd12259965788428922422362327131290619117264760597569863680;
    end else if ((1'b1 == ap_CS_fsm_state7)) begin
        cls_max_0_fu_82 <= or_ln203_reg_385;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        p_0_fu_74 <= 256'd0;
    end else if (((1'b1 == ap_CS_fsm_state5) & (grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out_o_ap_vld == 1'b1))) begin
        p_0_fu_74 <= grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out_o;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        phi_mul_reg_114 <= add_ln196_1_reg_333;
    end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2)) & (icmp_ln195_fu_162_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        phi_mul_reg_114 <= 8'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        t_reg_103 <= add_ln196_reg_341;
    end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2)) & (icmp_ln195_fu_162_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        t_reg_103 <= 4'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        vocabt_fu_70 <= 13'd0;
    end else if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln196_fu_184_p2 == 1'd1))) begin
        vocabt_fu_70 <= add_ln195_reg_328;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        add_ln195_reg_328 <= add_ln195_fu_168_p2;
        vocabt_1_reg_320 <= vocabt_fu_70;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        add_ln196_1_reg_333 <= add_ln196_1_fu_178_p2;
        add_ln196_reg_341 <= add_ln196_fu_190_p2;
        empty_33_reg_365 <= empty_33_fu_229_p2;
        empty_34_reg_370 <= empty_34_fu_235_p1;
        tmp_s_reg_349[7 : 5] <= tmp_s_fu_203_p3[7 : 5];
        trunc_ln203_reg_360 <= trunc_ln203_fu_221_p1;
        zext_ln203_reg_354[7 : 0] <= zext_ln203_fu_211_p1[7 : 0];
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        empty_35_reg_380 <= empty_35_fu_243_p2;
        gemm_stream_read_reg_375 <= gemm_stream_TDATA;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        or_ln203_reg_385 <= or_ln203_fu_277_p2;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2))) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((gemm_stream_TVALID == 1'b0)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_done == 1'b0)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

assign ap_ST_fsm_state8_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state8) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
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
    if ((1'b1 == ap_CS_fsm_state8)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln195_fu_162_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        cls_stream_TDATA_blk_n = cls_stream_TREADY;
    end else begin
        cls_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2)) & (icmp_ln195_fu_162_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        cls_stream_TVALID = 1'b1;
    end else begin
        cls_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((gemm_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
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
            if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2)) & (icmp_ln195_fu_162_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2)) & (icmp_ln195_fu_162_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln196_fu_184_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((gemm_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            if (((grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state5))) begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln195_fu_168_p2 = (vocabt_fu_70 + 13'd1);

assign add_ln196_1_fu_178_p2 = (phi_mul_reg_114 + 8'd23);

assign add_ln196_fu_190_p2 = (t_reg_103 + 4'd1);

assign and_ln203_fu_272_p2 = (xor_ln203_fu_266_p2 & cls_max_0_fu_82);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

always @ (*) begin
    ap_block_state2 = ((icmp_ln195_fu_162_p2 == 1'd1) & (cls_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln195_fu_162_p2 == 1'd1) & (cls_stream_TREADY == 1'b0));
end

assign cls_stream_TDATA = p_0_fu_74;

assign empty_33_fu_229_p2 = 256'd4294967295 << p_cast17_fu_225_p1;

assign empty_34_fu_235_p1 = empty_33_fu_229_p2[239:0];

assign empty_35_fu_243_p2 = (empty_33_reg_365 ^ 256'd115792089237316195423570985008687907853269984665640564039457584007913129639935);

assign empty_fu_199_p1 = t_reg_103[2:0];

assign grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start = grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_ap_start_reg;

assign icmp_ln195_fu_162_p2 = ((vocabt_fu_70 == 13'd6160) ? 1'b1 : 1'b0);

assign icmp_ln196_fu_184_p2 = ((t_reg_103 == 4'd8) ? 1'b1 : 1'b0);

assign lshr_ln203_fu_215_p2 = cls_max_0_fu_82 >> zext_ln203_fu_211_p1;

assign or_ln203_fu_277_p2 = (shl_ln203_1_fu_261_p2 | and_ln203_fu_272_p2);

assign p_cast17_fu_225_p1 = tmp_s_fu_203_p3;

assign shl_ln203_1_fu_261_p2 = zext_ln203_1_fu_257_p1 << zext_ln203_reg_354;

assign shl_ln203_fu_252_p2 = 184'd8388607 << zext_ln203_reg_354;

assign tmp_s_fu_203_p3 = {{empty_fu_199_p1}, {5'd0}};

assign trunc_ln203_fu_221_p1 = lshr_ln203_fu_215_p2[22:0];

assign xor_ln203_fu_266_p2 = (shl_ln203_fu_252_p2 ^ 184'd24519928653854221733733552434404946937899825954937634815);

assign zext_ln203_1_fu_257_p1 = grp_demux_llm_cls_Pipeline_VITIS_LOOP_199_3_fu_125_p_out1;

assign zext_ln203_fu_211_p1 = phi_mul_reg_114;

always @ (posedge ap_clk) begin
    tmp_s_reg_349[4:0] <= 5'b00000;
    zext_ln203_reg_354[183:8] <= 176'b00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
end

endmodule //DEMUX_demux_llm_cls
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module DEMUX_flow_control_loop_pipe_sequential_init(
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

// if no ap_continue port and current module is not DEMUX module, 
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

// if no ap_continue port and current module is not DEMUX module, ap_done handshakes with ap_start
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

module DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gemm_stream_TVALID,
        qk_stream_TREADY,
        gemm_stream_TDATA,
        gemm_stream_TREADY,
        qk_stream_TDATA,
        qk_stream_TVALID
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
input   gemm_stream_TVALID;
input   qk_stream_TREADY;
input  [239:0] gemm_stream_TDATA;
output   gemm_stream_TREADY;
output  [183:0] qk_stream_TDATA;
output   qk_stream_TVALID;

reg ap_idle;
reg gemm_stream_TREADY;
reg qk_stream_TVALID;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln130_fu_99_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln130_reg_216;
wire   [0:0] icmp_ln130_reg_216_pp0_iter0_reg;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gemm_stream_TDATA_blk_n;
reg    qk_stream_TDATA_blk_n;
reg   [22:0] trunc_ln4_reg_220;
reg   [22:0] trunc_ln139_1_reg_225;
reg   [22:0] trunc_ln139_2_reg_230;
reg   [22:0] trunc_ln139_3_reg_235;
reg   [22:0] trunc_ln139_4_reg_240;
reg   [22:0] trunc_ln139_5_reg_245;
reg   [22:0] trunc_ln139_6_reg_250;
reg   [22:0] trunc_ln139_7_reg_255;
reg   [7:0] indvar_flatten24_fu_74;
wire   [7:0] add_ln130_fu_105_p2;
wire    ap_loop_init;
reg   [7:0] ap_sig_allocacmp_indvar_flatten24_load;
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
//#0 indvar_flatten24_fu_74 = 8'd0;
//#0 ap_done_reg = 1'b0;
end

DEMUX_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        if ((icmp_ln130_fu_99_p2 == 1'd0)) begin
            indvar_flatten24_fu_74 <= add_ln130_fu_105_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten24_fu_74 <= 8'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln130_reg_216 <= icmp_ln130_fu_99_p2;
        trunc_ln139_1_reg_225 <= {{gemm_stream_TDATA[55:33]}};
        trunc_ln139_2_reg_230 <= {{gemm_stream_TDATA[85:63]}};
        trunc_ln139_3_reg_235 <= {{gemm_stream_TDATA[115:93]}};
        trunc_ln139_4_reg_240 <= {{gemm_stream_TDATA[145:123]}};
        trunc_ln139_5_reg_245 <= {{gemm_stream_TDATA[175:153]}};
        trunc_ln139_6_reg_250 <= {{gemm_stream_TDATA[205:183]}};
        trunc_ln139_7_reg_255 <= {{gemm_stream_TDATA[235:213]}};
        trunc_ln4_reg_220 <= {{gemm_stream_TDATA[25:3]}};
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln130_fu_99_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten24_load = 8'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten24_load = indvar_flatten24_fu_74;
    end
end

always @ (*) begin
    if (((icmp_ln130_fu_99_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        gemm_stream_TDATA_blk_n = gemm_stream_TVALID;
    end else begin
        gemm_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (icmp_ln130_fu_99_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        gemm_stream_TREADY = 1'b1;
    end else begin
        gemm_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln130_reg_216 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        qk_stream_TDATA_blk_n = qk_stream_TREADY;
    end else begin
        qk_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln130_reg_216 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        qk_stream_TVALID = 1'b1;
    end else begin
        qk_stream_TVALID = 1'b0;
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
            end else if (((~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (icmp_ln130_reg_216_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2)) | (~((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0)))) begin
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

assign add_ln130_fu_105_p2 = (ap_sig_allocacmp_indvar_flatten24_load + 8'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((icmp_ln130_fu_99_p2 == 1'd0) & (gemm_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln130_reg_216 == 1'd0) & (qk_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = ((icmp_ln130_reg_216 == 1'd0) & (qk_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_condition_63 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & ((1'b1 == ap_block_state2_io) | (1'b1 == ap_block_state2_pp0_stage0_iter1)))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln130_fu_99_p2 = ((ap_sig_allocacmp_indvar_flatten24_load == 8'd192) ? 1'b1 : 1'b0);

assign icmp_ln130_reg_216_pp0_iter0_reg = icmp_ln130_reg_216;

assign qk_stream_TDATA = {{{{{{{{trunc_ln139_7_reg_255}, {trunc_ln139_6_reg_250}}, {trunc_ln139_5_reg_245}}, {trunc_ln139_4_reg_240}}, {trunc_ln139_3_reg_235}}, {trunc_ln139_2_reg_230}}, {trunc_ln139_1_reg_225}}, {trunc_ln4_reg_220}};

endmodule //DEMUX_demux_llm_decoder_Pipeline_VITIS_LOOP_130_8_VITIS_LOOP_131_9_VITIS_LOOP_132_10
