
module mem_async_read #(
  parameter  width_p,
  parameter  depth_p, // depth and width_p can not be 0
  localparam addr_width_lp = $clog2(depth_p) 
) (
  input                       w_clk_i,
  input                       w_rst_ni,

  input                       w_v_i,
  input  [addr_width_lp-1:0]  w_addr_i,
  input  [width_p-1:0]        w_data_i,

  // currently unused
  input                       r_v_i,
  input  [addr_width_lp-1:0]  r_addr_i,

  output [width_p-1:0]        r_data_o
);

  // wire unused0 = w_rst_ni;
  wire unused1 = r_v_i;

  logic [width_p-1:0] mem [depth_p-1:0];

  // async read
  assign r_data_o = mem[r_addr_i];

  always_ff @(posedge w_clk_i or negedge w_rst_ni) begin
    if (~w_rst_ni) begin
      for (integer j = 0; j < depth_p; j++) begin
        mem[j] <= width_p'(0);
      end
    end
    else begin
      if (w_v_i) mem[w_addr_i] <= w_data_i;
    end
  end

    
endmodule