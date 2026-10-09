module gelu_top # (
    parameter DATA_WIDTH    = 16,
    parameter FRAC_WIDTH    = 12 
)(
    clk,
    rst_n,
    din,
    start,
    dout,
    busy,
    done
);

input                            clk;
input                            rst_n;
input       [DATA_WIDTH -1 : 0]  din;
input                            start;
output reg  [DATA_WIDTH -1 : 0]  dout;
output reg                       busy;
output reg                       done;

wire is_input_zero;
wire is_input_underflow;
wire is_input_overflow;
wire is_gelu_need;
wire [DATA_WIDTH -1 : 0] dout_w;

wire [15 : 0] gelu_din;
wire          gelu_start;
wire [15 : 0] gelu_dout;
wire          gelu_busy;
wire          gelu_done;

assign is_input_zero = din == {DATA_WIDTH{1'b0}};
assign is_input_underflow = din[DATA_WIDTH -1] & ~(&din[DATA_WIDTH - 1 : FRAC_WIDTH + 2] == 1'b1);
assign is_input_overflow = ~din[DATA_WIDTH -1] & ~(|din[DATA_WIDTH - 1 : FRAC_WIDTH + 2] == 1'b0);
assign is_gelu_need = ~(is_input_zero | is_input_underflow | is_input_overflow);

assign dout_w = is_input_zero ? {DATA_WIDTH{1'b0}} :
                is_input_overflow ? din :
                is_input_underflow ? {DATA_WIDTH{1'b0}} :
                {DATA_WIDTH{1'b0}};


always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        dout <= {DATA_WIDTH{1'b0}};
        busy <= 1'b0;
        done <= 1'b0;
    end
    else if (start & ~busy & ~is_gelu_need) begin
        dout <= dout_w;
        busy <= 1'b0;
        done <= 1'b1;
    end
    else if (start & ~busy & is_gelu_need) begin
        dout <= dout;
        busy <= 1'b1;
        done <= 1'b0;
    end
    else if (gelu_done) begin
        dout <= {{12-FRAC_WIDTH{gelu_dout[15]}}, gelu_dout[15:12-FRAC_WIDTH]};
        busy <= 1'b0;
        done <= 1'b1;
    end
    else begin
        dout <= dout;
        busy <= busy;
        done <= 1'b0;
    end
end

assign gelu_din = {din[FRAC_WIDTH + 3 : 0], {12-FRAC_WIDTH{1'b0}}};
assign gelu_start = start & is_gelu_need;

gelu #(
    .DATA_WIDTH(16),
    .FRAC_WIDTH(12) 
) gelu_inst (
    .clk(clk),
    .rst_n(rst_n),
    .din(gelu_din),
    .start(gelu_start),
    .dout(gelu_dout),
    .busy(gelu_busy),
    .done(gelu_done)
);

endmodule