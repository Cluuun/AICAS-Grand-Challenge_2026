module lod
#(
    parameter GROUP_SIZE =32,
    parameter SHIFT_DATA_WIDTH = 8,
    parameter DATA_WIDTH =16,
    parameter IN_WIDTH = 18,
    localparam IDX_WIDTH  = $clog2(IN_WIDTH)
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    input  logic [IN_WIDTH-1:0]   data_in,
    output logic [DATA_WIDTH-1:0] code,
    output logic                  out_valid
);
localparam FRAC_WIDTH = SHIFT_DATA_WIDTH-1;
localparam ALPHA = 5;
logic [IDX_WIDTH-1:0] pos;
logic [ALPHA-1:0]de_in;
logic [DATA_WIDTH-1:0]de_out;
logic [IN_WIDTH+ALPHA-1:0]expand_data_in;
assign expand_data_in = {data_in,{(ALPHA){1'b0}}};
assign de_in = expand_data_in[pos+ALPHA-1 -:ALPHA];
    always_comb begin 
        pos = '0;

        for(int i=0;i<GROUP_SIZE;i++)begin
            if(data_in[i])begin
                pos   = IDX_WIDTH'(i);
            end
        end
    end
    always_ff @( posedge clk or negedge rst_n ) begin
        if(rst_n==0)begin
            code   <='0;
        end
        else if(in_valid)begin
            code   <=(1+pos>SHIFT_DATA_WIDTH)?(de_out>>(1+pos-SHIFT_DATA_WIDTH)):(de_out<<(SHIFT_DATA_WIDTH-1-pos));
        end
        else begin
            code   <=code;
        end
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(rst_n==0)
        out_valid<=1'b0;
        else
        out_valid<=in_valid;
    end
    // always_comb begin
    // case (de_in)
    //     3'b000:de_out = 16'b0111100010011100;
    //     3'b001:de_out = 16'b0110101111100011;
    //     3'b010:de_out = 16'b0110000110011000;
    //     3'b011:de_out = 16'b0101100100011001;
    //     3'b100:de_out = 16'b0101000111110110;
    //     3'b101:de_out = 16'b0100101111100010;
    //     3'b110:de_out = 16'b0100011010100110;
    //     3'b111:de_out = 16'b0100001000010110;
    // endcase
    // end
always_comb begin
    case (de_in)
        5'd0 : de_out = 16'b0111111000001010; // 0x7e0a (Real: 0.984693)
        5'd1 : de_out = 16'b0111101001000111; // 0x7a47 (Real: 0.955295)
        5'd2 : de_out = 16'b0111011010111100; // 0x76bc (Real: 0.927601)
        5'd3 : de_out = 16'b0111001101100011; // 0x7363 (Real: 0.901468)
        5'd4 : de_out = 16'b0111000000111010; // 0x703a (Real: 0.876767)
        5'd5 : de_out = 16'b0110110100111100; // 0x6d3c (Real: 0.853384)
        5'd6 : de_out = 16'b0110101001100101; // 0x6a65 (Real: 0.831216)
        5'd7 : de_out = 16'b0110011110110100; // 0x67b4 (Real: 0.810170)
        5'd8 : de_out = 16'b0110010100100100; // 0x6524 (Real: 0.790164)
        5'd9 : de_out = 16'b0110001010110100; // 0x62b4 (Real: 0.771122)
        5'd10: de_out = 16'b0110000001100010; // 0x6062 (Real: 0.752976)
        5'd11: de_out = 16'b0101111000101010; // 0x5e2a (Real: 0.735665)
        5'd12: de_out = 16'b0101110000001100; // 0x5c0c (Real: 0.719131)
        5'd13: de_out = 16'b0101101000000111; // 0x5a07 (Real: 0.703325)
        5'd14: de_out = 16'b0101100000010111; // 0x5817 (Real: 0.688199)
        5'd15: de_out = 16'b0101011000111100; // 0x563c (Real: 0.673709)
        5'd16: de_out = 16'b0101010001110101; // 0x5475 (Real: 0.659817)
        5'd17: de_out = 16'b0101001011000000; // 0x52c0 (Real: 0.646487)
        5'd18: de_out = 16'b0101000100011101; // 0x511d (Real: 0.633684)
        5'd19: de_out = 16'b0100111110001001; // 0x4f89 (Real: 0.621379)
        5'd20: de_out = 16'b0100111000000101; // 0x4e05 (Real: 0.609542)
        5'd21: de_out = 16'b0100110010010000; // 0x4c90 (Real: 0.598148)
        5'd22: de_out = 16'b0100101100101000; // 0x4b28 (Real: 0.587172)
        5'd23: de_out = 16'b0100100111001110; // 0x49ce (Real: 0.576592)
        5'd24: de_out = 16'b0100100001111111; // 0x487f (Real: 0.566386)
        5'd25: de_out = 16'b0100011100111101; // 0x473d (Real: 0.556536)
        5'd26: de_out = 16'b0100011000000101; // 0x4605 (Real: 0.547022)
        5'd27: de_out = 16'b0100010011011000; // 0x44d8 (Real: 0.537828)
        5'd28: de_out = 16'b0100001110110100; // 0x43b4 (Real: 0.528938)
        5'd29: de_out = 16'b0100001010011010; // 0x429a (Real: 0.520337)
        5'd30: de_out = 16'b0100000110001010; // 0x418a (Real: 0.512011)
        5'd31: de_out = 16'b0100000010000001; // 0x4081 (Real: 0.503947)
        default: de_out = '0;
    endcase
end
endmodule