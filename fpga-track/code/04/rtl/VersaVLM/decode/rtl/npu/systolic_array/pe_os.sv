
module pe_os #(
  parameter DATA_WIDTH_IN   = 8, // 
  parameter DATA_WIDTH_OUT  = 32  //change it based on the number of rows with weight
) (
  input  logic                      clk_i,
  input  logic                      rstn_i,

  input  logic                      clear_i,   //clear_i=1,set mac_res_o to 0


  input  logic signed [DATA_WIDTH_IN-1:0]   weight_i,
  input  logic signed [DATA_WIDTH_IN-1:0]   active_i,
  
  output logic signed [DATA_WIDTH_IN-1:0]   weight_o,
  output logic signed [DATA_WIDTH_IN-1:0]   active_o,
  output logic signed [DATA_WIDTH_OUT-1:0]  mac_res_o
);


 // logic signed [2*DATA_WIDTH_IN-1:0] mul_res_w;
// Flow active & weight
 // assign mul_res_w = active_i * weight_i;

  always @(posedge clk_i or negedge rstn_i) begin
    if (~rstn_i) begin
      active_o <= '0;
      weight_o <= '0;
    end else begin
      active_o <= active_i;
      weight_o <= weight_i; 
    end

  end

  


  // Adder
  //logic signed [DATA_WIDTH_OUT-1:0] mac_res_w;

  
  //assign mac_res_w =mac_res_o  + mul_res_w;
  
  // Hold output
  always @(posedge clk_i or negedge rstn_i) begin
    if      (~rstn_i) 
            mac_res_o <= '0;
    else if (clear_i) 
            mac_res_o <= '0;
    else               
            mac_res_o <= mac_res_o + active_i * weight_i ;
  end
  


  
endmodule