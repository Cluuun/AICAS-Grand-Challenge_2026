
module arb_round_robin #(
  parameter width_p = 2
) (
  input                clk_i,
  input                rst_ni,

  input  [width_p-1:0] reqs_i,   // 
  output [width_p-1:0] grants_o, // one-hot selected item
  input                read_i
);

  if (width_p == 1) assign grants_o = reqs_i;
  else begin: arb
    // logic [width_p-1-1:0] thermocode_r, thermocode_n; 
    logic [width_p-1-1:0] thermocode_r;
    logic [width_p-1:0] thermocode_w; 

    logic [width_p-1:0] scan_in, scan_out;
    logic [$clog2(width_p):0][width_p-1:0] scan_t;

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (~rst_ni)
        thermocode_r <= '0; // initialize thermometer to all 0's
      else if (read_i)
        thermocode_r <= scan_out[width_p-1:1];
    end

    assign thermocode_w = (reqs_i & {1'b0,thermocode_r}) ? {1'b0,thermocode_r} : {width_p{1'b1}};

    assign scan_in = reqs_i & thermocode_w;
    assign scan_t[0] = scan_in;
    assign scan_out  = scan_t[$clog2(width_p)];

    for (genvar i = 0; i < $clog2(width_p); i=i+1) begin: scan
      wire [width_p-1:0] scan_shift = width_p'({ width_p'(0), scan_t[i] } >> (1 << i));
      assign scan_t[i+1] = scan_t[i] | scan_shift;
    end
  
    wire [width_p-1:0] detect_first = ~(scan_out >> 1) & scan_out;
    assign grants_o = detect_first & reqs_i;
  end


endmodule
