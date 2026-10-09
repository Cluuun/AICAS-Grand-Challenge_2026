`default_nettype none

// Fixed-depth valid/data delay line for SA edge skew.
//
// Reset behavior:
//   Only valid bits are reset. Data flops are intentionally not reset so the
//   block can map to SRL/register resources without unnecessary control sets.
module skew_delay_line #(
    parameter int DATA_W = 16,
    parameter int DELAY  = 0
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,
    input  wire logic              valid_i,
    input  wire logic [DATA_W-1:0] data_i,
    output logic                   valid_o,
    output logic [DATA_W-1:0]      data_o
);
    generate
        if (DELAY == 0) begin : gen_no_delay
            assign valid_o = valid_i;
            assign data_o  = data_i;
        end else begin : gen_delay
            logic [DELAY-1:0] valid_q;
            logic [DELAY-1:0][DATA_W-1:0] data_q;

            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    valid_q <= '0;
                end else begin
                    valid_q[0] <= valid_i;
                    data_q[0]  <= data_i;
                    for (int idx = 1; idx < DELAY; idx++) begin
                        valid_q[idx] <= valid_q[idx-1];
                        data_q[idx]  <= data_q[idx-1];
                    end
                end
            end

            assign valid_o = valid_q[DELAY-1];
            assign data_o  = data_q[DELAY-1];
        end
    endgenerate
endmodule

`default_nettype wire
