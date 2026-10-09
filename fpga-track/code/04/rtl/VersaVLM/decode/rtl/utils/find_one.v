// find the first one from LSB to MSB
module find_one #(
    parameter width = 4
)(
    input wire [width-1:0] din,
    output reg [$clog2(width)-1:0] sel
);

wire [width-1:0] tmp;
wire [width-1:0] onehot;

assign tmp = din - 1;
assign onehot = tmp ^ din;

integer i;
always@(*) begin
    for(i=0; i<width; i=i+1) begin
        if(onehot[i]) begin
            sel = i;
        end
    end
end
endmodule