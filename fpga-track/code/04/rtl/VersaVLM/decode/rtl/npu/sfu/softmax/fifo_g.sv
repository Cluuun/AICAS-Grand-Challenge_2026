module fifo_g
#(
    parameter DATA_WIDTH    = 16,
    parameter DEPTH         = 4
)
(
    input  wire                         clk_i,
    input  wire                         rst_n_i,
    // input channel
    input  wire     [DATA_WIDTH-1:0]    fifo_push_data_i,
    input  wire                         fifo_push_i,
    output wire                         fifo_ready_o,   //fifo is not full ,is ready for accept 
    // output channel
    output reg      [DATA_WIDTH-1:0]    fifo_pop_data_o,
    output wire                         fifo_valid_o,   //fifo is not empty
    input  wire                         fifo_pop_i
);
    
reg [DATA_WIDTH-1:0]       mem      [DEPTH-1:0] ;
reg [$clog2(DEPTH)-1:0]    push_ptr             ;
reg [$clog2(DEPTH)-1:0]    pop_ptr              ;
//wire [$clog2(DEPTH)-1:0]    pop_ptr_nxt           ;
reg [DEPTH  :0]            status_cnt           ;

assign fifo_valid_o = ~status_cnt[0];
assign fifo_ready_o = ~status_cnt[DEPTH];
//assign pop_ptr_nxt = fifo_pop_i?((pop_ptr + 'd1)==DEPTH?'d0:pop_ptr + 'd1):pop_ptr;
//Pointer update (one-hot shifting pointers)
always @ (posedge clk_i or negedge rst_n_i) begin: ff_push_ptr
    if (!rst_n_i) begin
        push_ptr <= 'd0;
    end else begin
        // fifo_push_i pointer
        if (fifo_push_i) begin
            push_ptr <= (push_ptr + 'd1)==DEPTH?'d0:push_ptr + 'd1;
        end
    end
end

always @ (posedge clk_i or negedge rst_n_i) begin: ff_pop_ptr
    if (!rst_n_i) begin
        pop_ptr <= 'd0;
    end else begin
        // fifo_pop_i pointer
        if (fifo_pop_i) begin
            pop_ptr <= (pop_ptr + 'd1)==DEPTH?'d0:pop_ptr + 'd1 ;
            //pop_ptr<=pop_ptr_nxt;
        end
    end
end
    
// Status (occupied slots) Counter
always @ (posedge clk_i or negedge rst_n_i) begin: ff_status_cnt
    if (!rst_n_i) begin
        status_cnt <= 1; // status counter onehot coded
    end else begin
        if (fifo_push_i & ~fifo_pop_i) begin
            // shift left status counter (increment)
            status_cnt <= { status_cnt[DEPTH-1:0],1'b0 } ;
        end else if (~fifo_push_i &  fifo_pop_i) begin
            // shift right status counter (decrement)
            status_cnt <= {1'b0, status_cnt[DEPTH:1] };
        end
    end
end
 
// data write (fifo_push_i) 
// address decoding needed for onehot fifo_push_i pointer
always @ (posedge clk_i or negedge rst_n_i) begin: ff_reg_dec
    if (!rst_n_i) begin
        for (int i=0; i<DEPTH; i=i+1) begin
            mem[i] <= '0;
        end
    end else if (fifo_push_i) begin
        mem[push_ptr] <= fifo_push_data_i;
    end
end

// always_comb begin : dataOut
    // fifo_pop_data_o = mem[pop_ptr];
// end
always_ff @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
        fifo_pop_data_o <= '0;
    end else begin
        //fifo_pop_data_o <= mem[pop_ptr_nxt];
        fifo_pop_data_o <= mem[pop_ptr];
    end
end
endmodule