module fifo_one_slot
#(
    parameter FULL_THROUGHPUT = 1'b0,
    parameter DATA_WIDTH      = 32,
    parameter GATING_FRIENDLY = 1'b1
)
(
    input   wire                   clk_i,
    input   wire                   rst_n_i,
    // input side
    input   wire                   valid_in,
    output  wire                   ready_out,
    input   wire[DATA_WIDTH-1:0]   data_in,
    
    //output side
    output  wire                   valid_out,
    input   wire                   ready_in,
    output  reg [DATA_WIDTH-1:0]   data_out
);
// ------------------------------------------------------------------------------------------------ //
reg                   full_r;
wire                  write_en;
// ------------------------------------------------------------------------------------------------ //
assign valid_out    = full_r;
assign ready_out    = write_en;

// write_en depends on impl.
generate
    if (FULL_THROUGHPUT) begin: gen_if_ft
        assign write_en = ready_in | ~full_r;
    end else begin: gen_if_ht
        assign write_en = ~full_r;
    end
endgenerate

// Full reg
always @(posedge clk_i or negedge rst_n_i) begin: ff_full_r
    if (!rst_n_i) begin
        full_r <= 0;
    end
    else begin
        if (write_en) begin
            full_r <= valid_in;
        end
    end
end

// data reg
always @(posedge clk_i or negedge rst_n_i) begin: ff_data_r
    if (!rst_n_i) begin
        data_out <= 'b0;
    end
    else begin
        if (write_en) begin
            if (valid_in) begin
                data_out <= data_in;
            end
        end
    end
end

endmodule
