////////////////////////////////////////////////////////////////////////////////////
//
// ZeroSoC version 1.2
//
// Module: ZeroCore -- div module
//
////////////////////////////////////////////////////////////////////////////////////

`define CORE_DATAWIDTH 32
`define CORE_NULLDATA 'b0

module div(
    // basic input
    input wire                          clk_i,
    input wire                          rst_n_i,

    // input from ex
    input wire                          start_i,
    input wire                          signed_flag_i,
    input wire [`CORE_DATAWIDTH-1:0]    dividend_i,
    input wire [`CORE_DATAWIDTH-1:0]    divisor_i,

    // output to ex
    output reg                          busy_o,
    output reg                          finish_o,
    output wire [`CORE_DATAWIDTH-1:0]   quotient_o,
    output wire [`CORE_DATAWIDTH-1:0]   remainder_o
);

    parameter IDLE    = 2'd0;
    parameter CAL     = 2'd1;
    parameter FIN     = 2'd2;
    
    reg [1:0] state;
    reg [4:0] counter;
    reg sign;
    reg [`CORE_DATAWIDTH-1:0] divisor_reg;
    reg [`CORE_DATAWIDTH-1:0] quotient_reg;
    reg [`CORE_DATAWIDTH-1:0] remainder_reg;
    wire [32:0] suboradd;
    wire [`CORE_DATAWIDTH-1:0] remainder_wire;
    wire didltdir;
    wire [`CORE_DATAWIDTH-1:0] dividend_abs;
    wire [`CORE_DATAWIDTH-1:0] divisor_abs;

    assign dividend_abs   = dividend_i[31] ? (~dividend_i+1) : dividend_i;
    assign divisor_abs    = divisor_i[31]  ? (~divisor_i+1) : divisor_i;
    assign didltdir       = signed_flag_i  ? (dividend_abs < divisor_abs) : (dividend_i < divisor_i);
    assign quotient_o     = (signed_flag_i & (divisor_i[31]^dividend_i[31]) & (dividend_i!=32'h80000000)) ? (~quotient_reg+1) : quotient_reg;
    assign remainder_wire = (sign & ~didltdir) ? (remainder_reg+divisor_reg) : remainder_reg;
    assign remainder_o    = (signed_flag_i & dividend_i[31] & ~didltdir) ? (~remainder_wire+1) : remainder_wire;
    assign suboradd       = sign ?    ({remainder_reg,quotient_reg[31]} + {1'b0,divisor_reg}) :
                                      ({remainder_reg,quotient_reg[31]} - {1'b0,divisor_reg}) ;
    
    always @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            busy_o <= 1'b0;
            finish_o <= 1'b0;
            state <= IDLE;
            counter <= 5'd0;
            sign <= 1'b0;
            divisor_reg <= `CORE_NULLDATA;
            quotient_reg <= `CORE_NULLDATA;
            remainder_reg <= `CORE_NULLDATA;
        end else begin
            case (state)
                IDLE: begin
                    if (start_i) begin
                        if (divisor_i == 32'd0) begin
                            busy_o <= 1'b0;
                            finish_o <= 1'b1;
                            divisor_reg <= 32'd0;
                            quotient_reg <= 32'hffffffff;
                            remainder_reg <= dividend_i;
                            counter <= 5'd0;
                            state <= IDLE;
                        end else begin
                            if (dividend_i == 32'h80000000 && divisor_i == 32'hffffffff && signed_flag_i) begin
                                busy_o <= 1'b0;
                                finish_o <= 1'b1;
                                divisor_reg <= 32'd0;
                                quotient_reg <= 32'h80000000;
                                remainder_reg <= 32'd0;
                                counter <= 5'd0;
                                state <= IDLE;
                            end else begin
                                if (didltdir) begin
                                    busy_o <= 1'b1;
                                    finish_o <= 1'b1;
                                    divisor_reg <= 32'd0;
                                    quotient_reg <= 32'd0;
                                    remainder_reg <= dividend_i;
                                    counter <= 5'd0;
                                    state <= IDLE;
                                end else begin
                                    busy_o <= 1'b1;
                                    finish_o <= 1'b0;
                                    divisor_reg <= (signed_flag_i & divisor_i[31]) ? (~divisor_i+1) : divisor_i;
                                    quotient_reg <= (signed_flag_i & dividend_i[31]) ? (~dividend_i+1) : dividend_i;
                                    remainder_reg <= 32'd0;
                                    counter <= 5'd0;
                                    state <= CAL;
                                end
                            end
                        end
                    end else begin
                        busy_o <= 1'b0;
                        finish_o <= 1'b0;
                        counter <= 5'd0;
                        sign <= 1'b0;
                        state <= IDLE;
                    end
                end 
                CAL: begin
                    busy_o <= 1'b1;
                    finish_o <= 1'b0;
                    remainder_reg <= suboradd[31:0];
                    sign <= suboradd[32];
                    quotient_reg <= {quotient_reg[30:0], ~suboradd[32]};
                    counter <= counter + 5'd1;
                    if (counter == 5'd30) begin
                        state <= FIN;
                    end else begin
                        state <= CAL;
                    end
                end 
                FIN: begin
                    remainder_reg <= suboradd[31:0];
                    sign <= suboradd[32];
                    quotient_reg <= {quotient_reg[30:0], ~suboradd[32]};
                    busy_o <= 1'b0;
                    counter <= 5'd0;
                    state <= IDLE;
                    finish_o <= 1'b1;
                end 
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule // div