/*
Module      : matrix_stream_transpose
Description : This module handles transposing a matrix which is being streamed
              one compute chunk at a time.

*/

`timescale 1ns / 1ps




module matrix_stream_transpose_new#(
    // Max dimensions
    parameter MAX_DIM0 = 2048,
    parameter MAX_DIM1 = 2048,

    // Compute dimensions
    parameter COMPUTE_DIM0 = 32,
    parameter COMPUTE_DIM1 = 32,

    // Other params
    parameter DATA_WIDTH = 8 ,
    localparam DIM0_WIDTH = $clog2(MAX_DIM0),
    localparam DIM1_WIDTH = $clog2(MAX_DIM1),
    localparam IN_DEPTH_DIM0 = MAX_DIM0 / COMPUTE_DIM0,
    localparam IN_DEPTH_DIM1 = MAX_DIM1 / COMPUTE_DIM1,
    localparam OUT_DEPTH_DIM0 = IN_DEPTH_DIM1,
    localparam OUT_DEPTH_DIM1 = IN_DEPTH_DIM0,
    localparam IN_ROW_COUNTER_WIDTH = $clog2(IN_DEPTH_DIM1) > 1 ? $clog2(IN_DEPTH_DIM1) : 1,
    localparam IN_COL_COUNTER_WIDTH = $clog2(IN_DEPTH_DIM0) > 1 ? $clog2(IN_DEPTH_DIM0) : 1,
    localparam OUT_ROW_COUNTER_WIDTH = $clog2(OUT_DEPTH_DIM1) > 1 ? $clog2(OUT_DEPTH_DIM1) : 1,
    localparam OUT_COL_COUNTER_WIDTH = $clog2(OUT_DEPTH_DIM0) > 1 ? $clog2(OUT_DEPTH_DIM0) : 1
) (
    input logic clk,
    input logic rst,

    input  logic [DIM0_WIDTH:0] in_real_dim0,
    input  logic [DIM1_WIDTH:0] in_real_dim1,
    // In Matrix
    input  logic [COMPUTE_DIM0*COMPUTE_DIM1-1:0][DATA_WIDTH-1:0] in_data,
    input  logic                  in_valid,
    output logic                  in_ready,

    // Out Matrix
    output logic [COMPUTE_DIM0*COMPUTE_DIM1-1:0][DATA_WIDTH-1:0] out_data ,
    output logic                  out_valid,
    input  logic                  out_ready,

    output logic[IN_ROW_COUNTER_WIDTH-1:0]   in_row_count,
    output logic [IN_COL_COUNTER_WIDTH-1:0]  in_col_count,
    output logic [OUT_ROW_COUNTER_WIDTH-1:0] out_row_count,
    output logic [OUT_COL_COUNTER_WIDTH-1:0] out_col_count,
    output logic [IN_COL_COUNTER_WIDTH-1:0]in_real_depth0,
    output logic [IN_ROW_COUNTER_WIDTH-1:0]in_real_depth1
);

localparam COM_WIDTH0=$clog2(COMPUTE_DIM0);
localparam COM_WIDTH1=$clog2(COMPUTE_DIM1);
logic [OUT_ROW_COUNTER_WIDTH-1:0]out_real_depth1;
logic [OUT_COL_COUNTER_WIDTH-1:0]out_real_depth0;
always_comb begin 
  in_real_depth0 = in_real_dim0[DIM0_WIDTH:COM_WIDTH0]+(|in_real_dim0[COM_WIDTH0-1:0]);
  in_real_depth1 = in_real_dim1[DIM1_WIDTH:COM_WIDTH1]+(|in_real_dim1[COM_WIDTH1-1:0]);
  out_real_depth0 = in_real_depth1;
  out_real_depth1 = in_real_depth0;
  //in_ready = 1'b1;
end


transpose #(
      .WIDTH(DATA_WIDTH),
      .DIM0 (COMPUTE_DIM0),
      .DIM1 (COMPUTE_DIM1)
) transpose_inst (
      .in_data (in_data),
      .out_data(out_data)
);
always_ff @( posedge clk or posedge rst ) begin 
    if(rst)
    out_valid<=1'b0;
    else if(out_valid&&out_ready)
    out_valid<=1'b0;
    else if(in_valid)
    out_valid<=1'b1;
    else 
    out_valid<=out_valid;
end
assign in_ready = ~out_valid;
always_ff @( posedge clk or posedge rst ) begin 
  if(rst)begin
    in_row_count <=0;
    in_col_count <=0;
    out_row_count<=0;
    out_col_count<=0;
  end
  else begin
    //Increment input side counters
    if (in_valid && in_ready) begin
      if (in_row_count == in_real_depth1 - 1 && in_col_count == in_real_depth0 - 1) begin
        // End of matrix
        in_row_count <= 0;
        in_col_count <= 0;
      end 
      else if (in_col_count == in_real_depth0 - 1) begin
        // End of row
        in_row_count <= in_row_count + 1;
        in_col_count <= 0;
      end else begin
        // Increment col counter
        in_col_count <= in_col_count + 1;
      end
    end
    else begin
      in_row_count <= in_row_count;
      in_col_count <= in_col_count;
    end
    //Increment output side counters
    if (out_valid && out_ready) begin
      if (out_row_count == out_real_depth1 - 1 && out_col_count == out_real_depth0 - 1) begin
        // End of matrix
        out_row_count <= 0;
        out_col_count <= 0;
      end else if (out_row_count == out_real_depth1 - 1) begin
        // End of col
        out_col_count <= out_col_count + 1;
        out_row_count <= 0;
      end else begin
        // Increment row counter
        out_row_count <= out_row_count + 1;
      end
    end
    else begin
      out_row_count <=out_row_count;  
      out_col_count <=out_col_count;
    end

  end
end
endmodule