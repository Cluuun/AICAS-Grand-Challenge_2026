module fp_leading_zeros_and_shift #(
  parameter int unsigned P    = 27,
  parameter int unsigned E    = 8,
  parameter int unsigned PLOG = 4
) (
  input  logic [P:0]   frac,
  input  logic [E-1:0] exp,
  output logic [P:0]   frac_Norm,
  output logic [E-1:0] exp_Norm,
  output logic         underFlow
);

  logic [PLOG:0] leadZerosBin;
  logic [E-1:0]  exp_Norm_int;
  logic          isZ;

  always_comb begin
    int unsigned count;

    count = 0;
    if (frac[P-1:0] == '0) begin
      count = P;
    end else begin
      for (int i = P-1; i >= 0; i--) begin
        if (frac[i] == 1'b0) begin
          count++;
        end else begin
          break;
        end
      end
    end

    leadZerosBin = count[PLOG:0];
  end

  always_comb begin
    if (isZ) begin
      exp_Norm_int = '0;
    end else begin
      exp_Norm_int = exp - {{(E-(PLOG+1)){1'b0}}, leadZerosBin};
    end
  end

  assign isZ       = (frac[P-1:0] == '0);
  assign underFlow = (!isZ && (exp > {{(E-(PLOG+1)){1'b0}}, leadZerosBin})) ? 1'b0 : 1'b1;
  assign exp_Norm  = exp_Norm_int;

  always_comb begin
    logic [P:0] temp;
    logic [P:0] dtemp;

    temp  = frac;
    dtemp = frac;
    for (int i = PLOG; i >= 0; i--) begin
      int unsigned sh;
      sh = 1 << i;
      if (leadZerosBin[i]) begin
        dtemp = temp << sh;
      end else begin
        dtemp = temp;
      end
      temp = dtemp;
    end
    frac_Norm = dtemp;
  end

endmodule
