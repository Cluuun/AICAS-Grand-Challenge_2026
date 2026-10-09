
module systolic_array_os #(
  parameter PE_FPGA_DSP       = 0,
  parameter PE_DATA_WIDTH_IN  = 8,
  parameter PE_DATA_WIDTH_OUT = 32,
  parameter ARRAY_WIDTH       = 16,
  parameter ARRAY_HEIGHT      = 16,
  parameter RF_DATA_WIDTH     = 64,
  parameter integer DSP_PE_NUM = (PE_FPGA_DSP ? ARRAY_WIDTH * ARRAY_HEIGHT : 0)
) (
  input  logic                        clk_i,
  input  logic                        rstn_i,
  
  input  logic                        ctrl_start_i,
  
  input  logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0] active_i,
  input  logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0]  weight_i,
  
  output logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_OUT-1:0] result_o_w 
);

  // PE array input & output
  logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] active_i_w ;
  logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] weight_i_w ;

  logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] active_o_w ;
  logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] weight_o_w ;

  localparam integer TOTAL_PE_NUM      = ARRAY_WIDTH * ARRAY_HEIGHT;
  localparam integer DSP_PE_NUM_CLAMPED = (DSP_PE_NUM < 0) ? 0 :
                                          ((DSP_PE_NUM > TOTAL_PE_NUM) ? TOTAL_PE_NUM : DSP_PE_NUM);

  // Generate PE array
  genvar i, j;
  generate
    for (i = 0; i < ARRAY_HEIGHT; i = i + 1) begin: sys_array_row
      for (j = 0; j < ARRAY_WIDTH; j = j + 1) begin: sys_array_col
        // Row-major mapping: the first DSP_PE_NUM PEs use DSP implementation.
        if ((i * ARRAY_WIDTH + j) < DSP_PE_NUM_CLAMPED) begin : gen_pe_fpga
            pe_os_fpga #(
              .DATA_WIDTH_IN  ( PE_DATA_WIDTH_IN  ),
              .DATA_WIDTH_OUT ( PE_DATA_WIDTH_OUT )
            ) u_pe_os (
              .clk_i       ( clk_i            ),
              .rstn_i      ( rstn_i           ),
            
              .clear_i     ( ctrl_start_i     ),
              
              .active_i    ( active_i_w[i][j] ),
              .weight_i    ( weight_i_w[i][j] ),
               
              .active_o    ( active_o_w[i][j] ),
              .weight_o    ( weight_o_w[i][j] ),
              .mac_res_o   ( result_o_w[i][j] )
            );
        end
        else begin : gen_pe_asic
            pe_os #(
              .DATA_WIDTH_IN  ( PE_DATA_WIDTH_IN  ),
              .DATA_WIDTH_OUT ( PE_DATA_WIDTH_OUT )
            ) u_pe_os (
              .clk_i       ( clk_i            ),
              .rstn_i      ( rstn_i           ),
            
              .clear_i     ( ctrl_start_i     ),
              
              .active_i    ( active_i_w[i][j] ),
              .weight_i    ( weight_i_w[i][j] ),
               
              .active_o    ( active_o_w[i][j] ),
              .weight_o    ( weight_o_w[i][j] ),
              .mac_res_o   ( result_o_w[i][j] )
            );
        end
      end
    end
  endgenerate

  // PE array connection
  // OS                                
  //            [ weight ]             
  //           ------------            
  // [active] | sys_array  |           
  //           ------------            

  generate
    for (i = 0; i < ARRAY_HEIGHT; i = i + 1) begin
      for (j = 0; j < ARRAY_WIDTH; j = j + 1) begin
        if ((i ==0) & (j == 0)) begin
          assign active_i_w[i][j] = active_i[i];
          assign weight_i_w[i][j] = weight_i[j];
        end else if (i == 0) begin
          assign active_i_w[i][j] = active_o_w[i][j-1];
          assign weight_i_w[i][j] = weight_i[j];
        end else if (j == 0) begin
          assign active_i_w[i][j] = active_i[i];
          assign weight_i_w[i][j] = weight_o_w[i-1][j];
        end else begin
          assign active_i_w[i][j] = active_o_w[i][j-1];
          assign weight_i_w[i][j] = weight_o_w[i-1][j];
        end
        //assign result_o[i] = result_o_w[ARRAY_HEIGHT-1][i];
        //assign result_o[i][j] = result_o_w[i][j];
      end
    end
  endgenerate
  
  // // debug
  // integer m, n;
  // always @(posedge clk_i) begin
  //   $display("active");
  //   for (m = 0; m < ARRAY_HEIGHT; m = m + 1) begin
  //     for (n = 0; n < ARRAY_WIDTH; n = n + 1) begin
  //       $write("[%0d][%0d] = %0d\t", m, n, active_i_w[m][n]);
  //     end
  //     $write("\n");
  //   end
  //   $write("\n\n");
  //   $display("weight");
  //   for (m = 0; m < ARRAY_HEIGHT; m = m + 1) begin
  //     for (n = 0; n < ARRAY_WIDTH; n = n + 1) begin
  //       $write("[%0d][%0d] = %0d\t", m, n, weight_i_w[m][n]);
  //     end
  //     $write("\n");
  //   end
  //   $write("\n\n");
  // end

endmodule
