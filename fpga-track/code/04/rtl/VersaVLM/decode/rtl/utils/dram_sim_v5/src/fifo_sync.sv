//
// fifo with 1 clock domain,
// using 1-write 1-async-read mem
// 
// input handshake: valid-and-ready or ready-then-valid
// output handshake: valid-then-ready (read after data-valid)
//

module fifo_sync #(
  parameter width_p,
  parameter depth_p,
  parameter ready_then_valid = 0,
  parameter ready_then_read  = 1
) (
  input                clk_i,
  input                rst_ni,

  input                valid_i,
  input  [width_p-1:0] data_i,
  output               ready_o,

  output               valid_o,
  output [width_p-1:0] data_o,
  input                read_i
);
  
  logic deque, enque;
  logic full, empty;

  if (ready_then_valid) begin
    assign enque = valid_i;
  end
  else begin
    assign enque = valid_i & ready_o;
  end

  if (ready_then_read) begin
    assign deque = read_i;
  end
  else begin
    assign deque = read_i & valid_o;
  end

if (depth_p > 1) begin: depth_gt_1
  localparam ptr_width_lp = $clog2(depth_p);
  logic [ptr_width_lp-1:0] wptr, rptr;
  logic [ptr_width_lp-1:0] wptr_add_1, rptr_add_1;
  logic equal_ptrs;
  logic enq_r, deq_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      enq_r <= 1'b0;
      deq_r <= 1'b1; // empty at first
    end
    else begin
      if (enque | deque) begin
        enq_r <= enque;
        deq_r <= deque;
      end
    end
  end

  assign wptr_add_1 = (wptr == depth_p - 1) ? ptr_width_lp'(0) : (wptr + 1'b1);
  assign rptr_add_1 = (rptr == depth_p - 1) ? ptr_width_lp'(0) : (rptr + 1'b1);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      wptr  <= ptr_width_lp'(0);
      rptr  <= ptr_width_lp'(0);
    end
    else begin
      wptr  <= enque ? wptr_add_1 : wptr;
      rptr  <= deque ? rptr_add_1 : rptr;
    end
  end

  assign equal_ptrs = rptr == wptr;
  assign empty = equal_ptrs & deq_r;
  assign full  = equal_ptrs & enq_r;

  mem_async_read #(
    .width_p ( width_p ),
    .depth_p ( depth_p )
  ) fifo_mem (
    .w_clk_i   ( clk_i   ),
    .w_rst_ni  ( rst_ni  ),

    .w_v_i     ( enque   ),
    .w_addr_i  ( wptr    ),
    .w_data_i  ( data_i  ),

    .r_v_i     ( valid_o ),
    .r_addr_i  ( rptr    ),
    .r_data_o  ( data_o  )
  );

  assign ready_o = ~full;
  assign valid_o = ~empty;
end // depth_gt_1

else begin: depth_eq_1
  logic [width_p-1:0] buffer;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
      buffer <= '0;
      full   <= 1'b0;
      empty  <= 1'b1;
    end
    else if (deque & enque) begin
      buffer <= data_i;
    end
    else if (deque) begin
      full   <= ~full;
      empty  <= ~empty;
    end
    else if (enque) begin
      buffer <= data_i;
      full   <= ~full;
      empty  <= ~empty;
    end
  end

  assign ready_o = ~full;
  assign valid_o = ~empty;
  assign data_o  = buffer;
end // depth_eq_1

  // synopsys translate_off
  always_ff @(posedge clk_i) begin
    if (rst_ni) begin
      assert ({empty, deque} !== 2'b11)
      else $error("invalid deque on empty fifo, %b, %b", empty, deque);

      if (ready_then_valid) begin
        assert ({full, enque, deque} !== 3'b110)
        else $error("invalid enque but no deque on full fifo");
      end
      // else no enque problem
      
      assert ({full, empty} !== 2'b11)
      else $error("fifo full and empty at the same time");
    end
  end
  // synopsys translate_on


endmodule
