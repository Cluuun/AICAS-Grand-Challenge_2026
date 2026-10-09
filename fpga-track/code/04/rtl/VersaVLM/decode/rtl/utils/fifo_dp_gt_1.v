//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab
// 
// Designer     : Haochen Xing
// Create Date  : 2020/11/26
// Project Name : ZeroSoC
// File Name    : fifo_dp_gt_1.v
//
// Description  : FIFO depth more than 1
//
// Revision: 
// Revision 1.0 - File Created
// 
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////

module FIFO_DP_GT_1 # (
    parameter DP = 4,
    parameter DW = 8
) (
    fifo_wen,
    fifo_wdat,
    fifo_full,
    fifo_ren,
    fifo_rdat,
    fifo_empty,
    clk,
    rst_n
);

input  fifo_wen;
input  fifo_wdat;
output fifo_full;
input  fifo_ren;
output fifo_rdat;
output fifo_empty;
input  clk;
input  rst_n;

// input signal declaration
wire          fifo_wen;
wire [DW-1:0] fifo_wdat;
wire          fifo_ren;
wire          clk;
wire          rst_n;

// output signal declaration
wire          fifo_full;
wire [DW-1:0] fifo_rdat;
wire          fifo_empty;

// inner signal declaration
reg fifo_full_r;
reg fifo_empty_r;

reg [DW-1:0]         mem [DP-1:0];  
reg [$clog2(DP)-1:0] wptr; // write pointer
reg [$clog2(DP)-1:0] rptr; // read pointer

reg almost_full;
reg almost_empty;

reg [$clog2(DP)-1:0] wptr_add_1;
reg [$clog2(DP)-1:0] rptr_add_1;

wire [1:0]  state;
integer j;

// circuit description
assign state = {fifo_wen, fifo_ren};

always @ (*) begin
    wptr_add_1   = (wptr == DP - 1) ? {($clog2(DP)){1'b0}} : (wptr + 1'b1);
    rptr_add_1   = (rptr == DP - 1) ? {($clog2(DP)){1'b0}} : (rptr + 1'b1);
    almost_full  = (wptr_add_1 == rptr);
    almost_empty = (rptr_add_1 == wptr);
end

always @ (posedge clk or negedge rst_n) begin
    if(rst_n == 1'b0) begin
        wptr         <= 0;   
        rptr         <= 0;   
        fifo_empty_r <= 1'b1;
        fifo_full_r  <= 1'b0;
        for(j = 0; j < DP; j = j + 1) begin
            mem[j] <= {DW{1'b0}};
        end
    end
    else begin
        case(state)
        2'b00: begin
            wptr         <= wptr;   
            rptr         <= rptr;   
            fifo_empty_r <= fifo_empty_r;
            fifo_full_r  <= fifo_full_r;
        end

        2'b10: begin //(wr_en && !rd_en) 
            if(almost_full) begin
                mem[wptr]    <= fifo_wdat;
                wptr         <= wptr_add_1;
                fifo_empty_r <= 1'b0;
                fifo_full_r  <= 1'b1;
            end
            else if(fifo_full_r == 1'b0) begin // ignore 'write' when full
                mem[wptr]    <= fifo_wdat;
                wptr         <= wptr_add_1;
                fifo_empty_r <= 1'b0;
                fifo_full_r  <= fifo_full_r;
            end
            else begin
                wptr         <= wptr;
                fifo_empty_r <= fifo_empty_r;
                fifo_full_r  <= fifo_full_r;
            end
        end

        2'b01: begin //(!wr_en && rd_en)
            if(almost_empty) begin
                rptr         <= rptr_add_1;
                fifo_empty_r <= 1'b1;
                fifo_full_r  <= 1'b0;
            end
            else if(fifo_empty_r == 1'b0) begin
                rptr         <= rptr_add_1;
                fifo_empty_r <= fifo_empty_r;
                fifo_full_r  <= 1'b0;
            end
            else begin
                rptr         <= rptr;
                fifo_empty_r <= fifo_empty_r;
                fifo_full_r  <= fifo_full_r;
            end
        end

        2'b11: begin //(wr_en && rd_en)
            if(fifo_empty_r) begin
                mem[wptr]    <= fifo_wdat;
                wptr         <= wptr_add_1;
                fifo_empty_r <= 1'b0;
            end
            else begin
                mem[wptr]    <= fifo_wdat;
                wptr         <= wptr_add_1;
                rptr         <= rptr_add_1;
            end   
        end
        endcase
    end 
end

assign fifo_full  = fifo_full_r;
assign fifo_empty = fifo_empty_r;
assign fifo_rdat  = mem[rptr];

endmodule