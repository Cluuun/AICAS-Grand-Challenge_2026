module right_shifter #(
  parameter int unsigned P    = 27,
  parameter int unsigned E    = 8,
  parameter int unsigned PLOG = 4
) (
  input  logic [P:0] frac,
  input  logic [E:0] diff_exp,
  output logic [P:0] frac_align
);

  logic [P:0] fracAlign_int;
  localparam logic [P:0] ALL_SHIFTED = {{P{1'b0}}, 1'b1};

  always_comb begin
    logic [P:0] temp;
    logic [P:0] dtemp;
    logic [P:0] zeros;
    logic       sticky;

    temp   = frac;
    dtemp  = frac;
    zeros  = '0;
    sticky = 1'b0;

    for (int i = PLOG; i >= 0; i--) begin
      int unsigned sh;
      logic lost_nonzero;
      sh = 1 << i;
      lost_nonzero = 1'b0;
      if (diff_exp[i]) begin
        dtemp = temp >> sh;
        for (int b = 0; b < sh; b++) begin
          lost_nonzero |= temp[b];
        end
        if (lost_nonzero) begin
          sticky = 1'b1;
        end else begin
          sticky = 1'b0;
        end
      end else begin
        dtemp = temp;
      end
      temp = dtemp;
    end

    fracAlign_int = {dtemp[P:1], (dtemp[0] | sticky)};
  end

  assign frac_align = (diff_exp[E:PLOG+1] == '0) ? fracAlign_int : ALL_SHIFTED;

endmodule
